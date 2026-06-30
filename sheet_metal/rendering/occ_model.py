"""
sheet_metal_occ.occ_model — pythonOCC backend with TRUE OCC B-Rep geometry.

Every shape is an exact OCC B-Rep object (no mesh wrapping):
  FlatPanel / PolygonPanel → single planar OCC face (arc corners via GC_MakeArcOfCircle)
#   SplinePanel              → single planar OCC face (Geom_BSplineCurve for curved edges)
  LoftPanel  → BRepOffsetAPI_ThruSections; aux-profile tangent steering when t_start/t_end set
  PipePanel  → BRepOffsetAPI_MakePipeShell (SetDiscreteMode + midpoint Location); falls back
                to ThruSections+aux on failure
                             Curve3D.line  → exact line-segment wire (BRepBuilderAPI_MakeEdge)
                             Curve3D.arc   → exact arc wire (GC_MakeArcOfCircle)
                             Curve3D.bspline/polyline → dense polygon wire approximation
  BendArc (two backends, selectable via arc_backend parameter):
    'loft' (default) — arc-guided ThruSections loft.  Handles non-aligned parent/
                        child edges: different lengths, non-parallel directions, B-splines.
                        Each corresponding pair (pA, cA) sweeps along a local arc defined
                        by T_out = (cA − C + r·cos α·N) / (r·sin α).
    'cyl'            — exact Geom_CylindricalSurface.  Aligned, equal-length, straight
                        edges only.
  PartialBendArc           → BRepOffsetAPI_ThruSections ruled loft (line-segment profiles)
#   CurvedBendArc            → arc-guided ThruSections loft (uses _build_grid child edge)
  StampedRecess            → BRepOffsetAPI_ThruSections loft (rounded-rect at each depth level)
                             + BRepBuilderAPI_MakeFace flat bottom
  PolygonRecess            → BRepOffsetAPI_ThruSections loft (offset polygon at each depth level)
                             + BRepBuilderAPI_MakeFace flat bottom

STEP export produces compact, exact B-Rep files.
"""

import numpy as np

from ..basics.maths import n
from .basics import BaseRenderer
from ..basics.panel import FlatPanel, PolygonPanel, LoftPanel, PipePanel
from ..basics.bending import BendArc, PartialBendArc
from ..basics.recess import StampedRecess, PolygonRecess, _sample_profile

# ── OCC imports ───────────────────────────────────────────────────────────────
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax3
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_BSplineCurve
from OCC.Core.TColgp import (TColgp_Array1OfPnt, TColgp_HArray1OfPnt,
                              TColgp_Array1OfVec)
from OCC.Core.TColStd import (TColStd_Array1OfReal, TColStd_Array1OfInteger,
                               TColStd_HArray1OfBoolean)
from OCC.Core.GeomAPI import GeomAPI_Interpolate, GeomAPI_ProjectPointOnCurve
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
)
from OCC.Core.BRepOffsetAPI import (BRepOffsetAPI_ThruSections,
                                    BRepOffsetAPI_MakePipeShell)
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone



# ─── geometry helpers ─────────────────────────────────────────────────────────

def _to_pnt(arr):
    a = np.asarray(arr, float)
    return gp_Pnt(float(a[0]), float(a[1]), float(a[2]))

def _to_dir(arr):
    a = n(np.asarray(arr, float))
    return gp_Dir(float(a[0]), float(a[1]), float(a[2]))


# ─── mesh fallback (used if OCC B-Rep building fails) ─────────────────────────

def _dedup_pts(pts, tol=1e-9):
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(np.asarray(p) - np.asarray(out[-1])) > tol:
            out.append(p)
    return out

def _wire_from_pts(pts):
    pts = _dedup_pts([np.asarray(p, float) for p in pts])
    if len(pts) < 3:
        return None
    poly = BRepBuilderAPI_MakePolygon()
    for p in pts:
        poly.Add(_to_pnt(p))
    poly.Close()
    return poly.Wire() if poly.IsDone() else None

def _compound_from_face_lists(faces_list):
    """OCC compound from polygon face list — fallback only."""
    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    added = 0
    for face_pts in faces_list:
        pts = [np.asarray(p, float) for p in face_pts]
        if len(pts) < 3:
            continue
        try:
            wire = _wire_from_pts(pts)
            if wire is None:
                continue
            maker = BRepBuilderAPI_MakeFace(wire, True)
            if maker.IsDone():
                builder.Add(compound, maker.Face())
                added += 1
        except Exception:
            pass
    return compound if added > 0 else None


# ─── polygon wire builder ─────────────────────────────────────────────────────

def _flat_panel_wire_occ(verts_2d, corner_r, blocked, origin, T, W):
    """Build an exact closed OCC wire for a 2-D flat panel.

    Corners between straight edges are rounded with GC_MakeArcOfCircle. 
    - we don't support B-spline edges

    Parameters
    ----------
    verts_2d    : (N, 2) CCW polygon vertices in panel local (u, v) frame
    corner_r    : rounding radius (0 = all sharp)
    blocked     : set of vertex indices that must stay sharp
    origin, T, W: 3-D panel frame  (3-D point = origin + u*T + v*W)
    """
    verts_2d = np.asarray(verts_2d, float)
    n        = len(verts_2d)
    origin   = np.asarray(origin, float)
    T        = np.asarray(T, float)
    W        = np.asarray(W, float)

    def to_3d(uv):
        uv = np.asarray(uv, float)
        return origin + float(uv[0]) * T + float(uv[1]) * W

    # ── Precompute arc geometry at each vertex ────────────────────────────────
    # Arc is inserted only where BOTH adjacent edges are straight (not B-spline)
    # and the corner is convex and not blocked.
    arcs = [None] * n   # (P_start_3d, P_mid_3d, P_end_3d) or None

    if corner_r > 1e-12:
        for i in range(n):
            if i in blocked:
                continue

            v      = verts_2d[i]
            v_prev = verts_2d[(i - 1) % n]
            v_next = verts_2d[(i + 1) % n]

            e_in  = v - v_prev
            e_out = v_next - v
            l_in  = np.linalg.norm(e_in)
            l_out = np.linalg.norm(e_out)
            if l_in < 1e-9 or l_out < 1e-9:
                continue
            d_in  = e_in / l_in
            d_out = e_out / l_out

            cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
            if cross < 1e-6:
                continue   # concave or collinear

            cos_a = np.clip(np.dot(d_in, d_out), -1.0, 1.0)
            sin_h = np.sqrt(max(0.0, (1 - cos_a) / 2))
            cos_h = np.sqrt(max(0.0, (1 + cos_a) / 2))
            if cos_h < 1e-9:
                continue

            trim = corner_r * sin_h / cos_h
            # Guard: trim must not exceed half the edge length
            if trim > l_in / 2 - 1e-6 or trim > l_out / 2 - 1e-6:
                continue

            arc_start_2d = v - trim * d_in
            arc_end_2d   = v + trim * d_out

            # Arc centre: rotate d_in by +90° (CCW) → inward perpendicular
            perp_in = np.array([-d_in[1], d_in[0]])
            C_2d    = arc_start_2d + corner_r * perp_in

            # Arc midpoint = C + r * normalise(v_start + v_end from C)
            vs_ = arc_start_2d - C_2d
            ve_ = arc_end_2d   - C_2d
            vm_ = vs_ + ve_
            vm_len = np.linalg.norm(vm_)
            if vm_len > 1e-9:
                arc_mid_2d = C_2d + corner_r * vm_ / vm_len
            else:
                arc_mid_2d = (arc_start_2d + arc_end_2d) / 2

            arcs[i] = (to_3d(arc_start_2d), to_3d(arc_mid_2d), to_3d(arc_end_2d))

    # ── Build OCC edges in CCW order ──────────────────────────────────────────
    # For each edge i → j:
    #   - If B-spline: add exact Geom_BSplineCurve edge (no arcs at its endpoints)
    #   - If straight: add line segment, then arc at vertex j
    edges = []

    for i in range(n):
        j     = (i + 1) % n

        seg_start = arcs[i][2] if arcs[i] is not None else to_3d(verts_2d[i])
        seg_end   = arcs[j][0] if arcs[j] is not None else to_3d(verts_2d[j])

        if np.linalg.norm(np.asarray(seg_end) - np.asarray(seg_start)) > 1e-9:
            edges.append(BRepBuilderAPI_MakeEdge(
                _to_pnt(seg_start), _to_pnt(seg_end)).Edge())

        # Arc at vertex j (only between two straight edges)
        if arcs[j] is not None:
            P1, Pm, P3 = arcs[j]
            arc = GC_MakeArcOfCircle(_to_pnt(P1), _to_pnt(Pm), _to_pnt(P3))
            if arc.IsDone():
                edges.append(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())

    if not edges: raise ValueError('No edges created')

    wb = BRepBuilderAPI_MakeWire()
    for e in edges:
        wb.Add(e)
        
    if not wb.IsDone(): raise ValueError('Wire build failed')
    return wb.Wire()


# ─── Curve3D → OCC wire ───────────────────────────────────────────────────────

def _curve3d_to_occ_wire(crv, n_poly=64):
    """Convert a Curve3D to an OCC wire.

    Curve3D.line    → exact single-edge wire (BRepBuilderAPI_MakeEdge p0→p1)
    Curve3D.arc     → exact arc wire         (GC_MakeArcOfCircle)
    Curve3D.bspline → exact Geom_BSplineCurve wire
    Curve3D.polyline → polygon wire from stored points
    """
    kind = crv._kind

    if kind == 'line':
        edge = BRepBuilderAPI_MakeEdge(_to_pnt(crv.p0), _to_pnt(crv.p1)).Edge()
        wb = BRepBuilderAPI_MakeWire(edge)
        if not wb.IsDone(): raise ValueError('Wire build failed')
        return wb.Wire()

    if kind == 'arc':
        # Sample 3 points: start, mid, end for GC_MakeArcOfCircle
        pts = crv._eval_raw(np.array([0.0, 0.5, 1.0]))
        arc_geom = GC_MakeArcOfCircle(
            _to_pnt(pts[0]), _to_pnt(pts[1]), _to_pnt(pts[2]))
        if not arc_geom.IsDone(): raise ValueError('Arc build failed')
        edge = BRepBuilderAPI_MakeEdge(arc_geom.Value()).Edge()
        wb = BRepBuilderAPI_MakeWire(edge)
        if not wb.IsDone(): raise ValueError('Wire build failed')
        return wb.Wire()

    if kind == 'bspline':
        cp   = crv._cp
        deg  = crv._deg
        knts = crv._knots
        n_p  = len(cp)
        distinct, mults = [], []
        for k in knts:
            if not distinct or abs(k - distinct[-1]) > 1e-12:
                distinct.append(float(k))
                mults.append(1)
            else:
                mults[-1] += 1
        n_k = len(distinct)
        poles_arr = TColgp_Array1OfPnt(1, n_p)
        knots_arr = TColStd_Array1OfReal(1, n_k)
        mults_arr = TColStd_Array1OfInteger(1, n_k)
        for ki, p in enumerate(cp):
            poles_arr.SetValue(ki + 1, _to_pnt(p))
        for ki, k in enumerate(distinct):
            knots_arr.SetValue(ki + 1, k)
        for ki, m in enumerate(mults):
            mults_arr.SetValue(ki + 1, m)
        bspl = Geom_BSplineCurve(poles_arr, knots_arr, mults_arr, deg)
        edge = BRepBuilderAPI_MakeEdge(bspl, 0.0, 1.0).Edge()
        wb = BRepBuilderAPI_MakeWire(edge)
        if not wb.IsDone(): raise ValueError('Wire build failed')
        return wb.Wire()

    # polyline / fallback: dense polygon wire
    if kind == 'polyline':
        pts = crv._pts
    pts = crv.sample(n_poly)
    poly = BRepBuilderAPI_MakePolygon()
    for p in pts:
        poly.Add(_to_pnt(p))
    if not poly.IsDone(): raise ValueError('Polygon build failed')
    wb = BRepBuilderAPI_MakeWire(poly.Wire())
    if not wb.IsDone(): raise ValueError('Wire build failed')
    return wb.Wire()


def _loft_panel_occ(panel: LoftPanel):
    """ThruSections loft with tangent-steering auxiliary profiles.

    Inserts a phantom profile δ past the root (in t_start direction) and δ
    before the free end (in −t_end direction).  The loft departs/arrives in
    the requested tangent direction (approximate G1).  Falls back to plain
    ThruSections when t_start/t_end are absent or on any failure.
    """
    profiles = list(panel.profiles)
    if len(profiles) < 2: raise ValueError('LoftPanel needs at least 2 profiles')

    def _shift_crv(crv, vec):
        a0, b0 = crv.endpoints()
        return type(crv).line(np.asarray(a0) + vec, np.asarray(b0) + vec)

    # tangent-steering auxiliary profiles
    t_start = getattr(panel, 't_start', None)
    t_end   = getattr(panel, 't_end',   None)

    if t_start is not None or t_end is not None:
        root_mid = np.asarray(panel.spine_pts[0], float)
        free_mid = np.asarray(panel.spine_pts[-1], float)
        arm_len  = float(np.linalg.norm(free_mid - root_mid))
        delta    = max(1.0, arm_len * 0.025)

        sections = [profiles[0]]
        if t_start is not None:
            sections.append(_shift_crv(profiles[0],  delta * n(t_start)))
        sections.extend(profiles[1:-1])
        if t_end is not None:
            sections.append(_shift_crv(profiles[-1], -delta * n(t_end)))
        sections.append(profiles[-1])
    else:
        sections = profiles

    thru = BRepOffsetAPI_ThruSections(False, panel.ruled)
    n_added = 0
    for crv in sections:
        w = _curve3d_to_occ_wire(crv)
        thru.AddWire(w)
        n_added += 1
    
    thru.Build()
    if not thru.IsDone(): raise ValueError('ThruSections build failed') 
    return thru.Shape()

def _pipe_panel_occ(panel: PipePanel):
    """BRepOffsetAPI_MakePipeShell with the correct calling convention.

    Key fixes vs. the original broken implementation:
    - SetDiscreteMode()  avoids Frenet-frame singularity on straight spines
    - Each profile wire is split at its midpoint (= the spine waypoint for
      bracket profiles); the midpoint vertex is passed as Location with
      WithContact=True so OCC places it exactly at the matching spine point
    - WithCorrection=True is NOT used: OCC rejects it for open/non-planar wires
    """
    profiles  = list(panel.profiles)
    spine_pts = [np.asarray(p, float) for p in panel.spine_pts]
    n_pts     = len(profiles)
    if n_pts < 2 or len(spine_pts) != n_pts: raise ValueError('PipePanel needs at least 2 profiles and spine points')

    # ── 1. Build spine BSpline with optional tangent constraints ──────────
    pts_arr = TColgp_HArray1OfPnt(1, n_pts)
    for i, pt in enumerate(spine_pts):
        pts_arr.SetValue(i + 1, _to_pnt(pt))
    interp = GeomAPI_Interpolate(pts_arr, False, 1e-7)

    t_start = getattr(panel, 't_start', None)
    t_end   = getattr(panel, 't_end',   None)
    if t_start is not None or t_end is not None:
        tang_arr = TColgp_Array1OfVec(1, n_pts)
        flag_arr = TColStd_HArray1OfBoolean(1, n_pts)
        for i in range(n_pts):
            flag_arr.SetValue(i + 1, False)
            tang_arr.SetValue(i + 1, gp_Vec(0.0, 0.0, 0.0))
        if t_start is not None:
            tang_arr.SetValue(1, gp_Vec(*[float(x) for x in t_start]))
            flag_arr.SetValue(1, True)
        if t_end is not None:
            tang_arr.SetValue(n_pts, gp_Vec(*[float(x) for x in t_end]))
            flag_arr.SetValue(n_pts, True)
        interp.Load(tang_arr, flag_arr)

    interp.Perform()
    if not interp.IsDone(): raise ValueError('Interpolation failed')
    bspline = interp.Curve()

    # ── 2. Project spine waypoints onto BSpline to get parameters ─────────
    params = []
    for pt in spine_pts:
        proj = GeomAPI_ProjectPointOnCurve(_to_pnt(pt), bspline)
        if proj.NbPoints() > 0:
            params.append(proj.LowerDistanceParameter())
        else:
            frac = len(params) / (n_pts - 1) if n_pts > 1 else 0.0
            p0_c, p1_c = bspline.FirstParameter(), bspline.LastParameter()
            params.append(p0_c + frac * (p1_c - p0_c))
    params.sort()

    # ── 3. Build spine wire ───────────────────────────────────────────────
    Vs = [BRepBuilderAPI_MakeVertex(bspline.Value(float(p))).Vertex()
            for p in params]
    wm = BRepBuilderAPI_MakeWire()
    for i in range(n_pts - 1):
        wm.Add(BRepBuilderAPI_MakeEdge(
            bspline, Vs[i], Vs[i + 1], params[i], params[i + 1]).Edge())
    if not wm.IsDone(): raise ValueError('Wire build failed')
    spine_wire = wm.Wire()

    # ── 4. Build pipe shell ───────────────────────────────────────────────
    pipe = BRepOffsetAPI_MakePipeShell(spine_wire)
    pipe.SetDiscreteMode()

    for crv in profiles:
        a0, b0 = crv.endpoints()
        a0, b0 = np.asarray(a0, float), np.asarray(b0, float)
        mid = (a0 + b0) / 2.0
        pwm = BRepBuilderAPI_MakeWire()
        pwm.Add(BRepBuilderAPI_MakeEdge(_to_pnt(a0),  _to_pnt(mid)).Edge())
        pwm.Add(BRepBuilderAPI_MakeEdge(_to_pnt(mid), _to_pnt(b0)).Edge())
        if not pwm.IsDone(): raise ValueError('Wire build failed')
        v_mid = BRepBuilderAPI_MakeVertex(_to_pnt(mid)).Vertex()
        pipe.Add(pwm.Wire(), v_mid, True, False)

    pipe.Build()
    if not pipe.IsDone(): raise ValueError('PipeShell build failed')
    return pipe.Shape()


# ─── bend arc builders ─────────────────────────────────────────────────────────

# ── Backend A: exact Geom_CylindricalSurface ─────────────────────────────────
# Assumes both edges are parallel, equal-length, and straight.  Exact for
# aligned BendArc; incorrect for partial-span, non-parallel, or curved edges.

def _bend_arc_cyl_occ(arc: BendArc):
    """Exact Geom_CylindricalSurface face for an aligned straight BendArc."""
    p0     = np.asarray(arc.p0, float)
    p1     = np.asarray(arc.p1, float)
    N_bend = n(np.asarray(arc.N_bend, float))
    alpha  = float(arc.alpha)
    r_n    = float(arc.r_n)

    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-12 or abs(alpha) < 1e-9:
        raise ValueError('BendArc is too short or has no angle')

    edge_dir = (p1 - p0) / L
    C0  = p0 + r_n * N_bend
    ax3 = gp_Ax3(_to_pnt(C0), _to_dir(edge_dir), _to_dir(-N_bend))
    cyl = Geom_CylindricalSurface(ax3, r_n)

    maker = BRepBuilderAPI_MakeFace(cyl, -alpha, 0.0, 0.0, L, 1e-6)
    if not maker.IsDone(): raise ValueError('MakeFace failed')
    return maker.Face()


# ── Backend B: arc-guided ThruSections loft ───────────────────────────────────
# Works for ANY pair of parent/child edge geometries: different lengths,
# non-parallel directions, B-spline curves.
#
# Key idea: for each corresponding pair of samples (pA_j, cA_j) on the parent
# and child edges, the local arc center and tangent-out are back-derived from:
#
#   C_j    = pA_j + r_n · N_bend
#   T_out_j = (cA_j − C_j + r_n·cos(α)·N_bend) / (r_n·sin(α))
#
# This satisfies pA_j = C_j − r_n·N_bend (at a=0) and cA_j (at a=α) exactly.
# At intermediate angle a: P(a) = C_j + r_n·(−cos(a)·N + sin(a)·T_out_j).
#
# For an aligned straight BendArc all T_out_j are identical to arc.T_out
# and the loft reduces to the cylindrical surface (within tessellation error).

def _arc_loft_occ(parent_pts, child_pts, N_bend, alpha, r_n, n_arc=24):
    """Arc-guided ThruSections loft between sampled parent and child edges.

    Parameters
    ----------
    parent_pts : (M+1, 3) 3-D samples along the parent (source) edge
    child_pts  : (M+1, 3) 3-D samples along the child  (target) edge
    N_bend     : unit bend normal
    alpha      : total bend angle (radians)
    r_n        : neutral-axis bend radius
    n_arc      : number of arc-angle sections (more → smoother curve)

    Returns OCC shape or None on failure.
    """
    Nb    = n(np.asarray(N_bend, float))
    sa_alpha = np.sin(float(alpha)) 
    ca_alpha = np.cos(float(alpha))
    if abs(sa_alpha) < 1e-9: raise ValueError('alpha is too small')

    parent_pts = np.asarray(parent_pts, float)
    child_pts  = np.asarray(child_pts,  float)

    # Arc centres along the parent edge
    C_js = parent_pts + r_n * Nb                           # (M+1, 3)

    # Local tangent-out back-derived from cA = C + r·(−cos α·Nb + sin α·T_out)
    T_outs = (child_pts - C_js + r_n * ca_alpha * Nb) / (r_n * sa_alpha)  # (M+1, 3)

    thru = BRepOffsetAPI_ThruSections(False, False)   # shell, smooth loft
    n_added = 0

    for a in np.linspace(0.0, float(alpha), n_arc + 1):
        ca, sa = np.cos(a), np.sin(a)
        # Profile points at this arc angle
        pts_k = C_js + r_n * (-ca * Nb + sa * T_outs)   # (M+1, 3)

        poly = BRepBuilderAPI_MakePolygon()
        for p in pts_k:
            poly.Add(_to_pnt(p))
        if poly.IsDone():
            thru.AddWire(poly.Wire())
            n_added += 1

    if n_added < 2:
        raise ValueError('Not enough wires added')
    
    thru.Build()
    if not thru.IsDone(): raise ValueError('Build failed')
    return thru.Shape()


def _bend_arc_loft_occ(arc: BendArc, n_arc=24):
    """Arc-guided loft for BendArc (straight, possibly partial-span)."""
    p0    = np.asarray(arc.p0, float)
    p1    = np.asarray(arc.p1, float)
    T_out = n(np.asarray(arc.T_out, float))
    Nb    = n(np.asarray(arc.N_bend, float))
    alpha = float(arc.alpha)
    r_n   = float(arc.r_n)

    if abs(alpha) < 1e-9 or np.linalg.norm(p1 - p0) < 1e-9:
        raise ValueError('BendArc is too short or has no angle')

    # For a straight edge just the two endpoints are needed
    parent_pts = np.array([p0, p1])
    C_pts      = parent_pts + r_n * Nb
    child_pts  = C_pts + r_n * (-np.cos(alpha) * Nb + np.sin(alpha) * T_out)

    return _arc_loft_occ(parent_pts, child_pts, Nb, alpha, r_n, n_arc=n_arc)


def _partial_arc_occ(arc: PartialBendArc, n_alpha=24):
    """ThruSections ruled loft for PartialBendArc — exact OCC surface."""
    p0    = np.asarray(arc.p0, float)
    p1    = np.asarray(arc.p1, float)
    T_out = n(np.asarray(arc.T_out, float))
    Nb    = n(np.asarray(arc.N_bend, float))
    alpha = float(arc.alpha)
    r_n   = float(arc.r_n)
    L     = float(arc.L)
    s0    = float(arc.s0)
    s1    = float(arc.s1)
    blend = float(arc.blend)
    taper = arc.taper

    if L < 1e-12 or abs(alpha) < 1e-9:
        raise ValueError('PartialBendArc is too short or has no angle')

    def pt_at(s, a):
        frac   = np.clip(s / L, 0.0, 1.0)
        p_edge = (1.0 - frac) * p0 + frac * p1
        C      = p_edge + r_n * Nb
        return C + r_n * (-np.cos(a) * Nb + np.sin(a) * T_out)

    def ext(a):
        t = a / alpha
        if taper == 'linear':
            return blend * (1.0 - t)
        return blend * (1.0 - np.sqrt(max(0.0, t * (2.0 - t))))

    a_vals = np.linspace(0.0, alpha, n_alpha + 1)

    thru = BRepOffsetAPI_ThruSections(False, True)   # shell, ruled
    for a in a_vals:
        e    = ext(a)
        s_lo = max(0.0, s0 - e)
        s_hi = min(L,   s1 + e)
        pA   = pt_at(s_lo, a)
        pB   = pt_at(s_hi, a)
        seg  = BRepBuilderAPI_MakeEdge(_to_pnt(pA), _to_pnt(pB)).Edge()
        w    = BRepBuilderAPI_MakeWire(seg).Wire()
        thru.AddWire(w)

    thru.Build()
    if not thru.IsDone(): raise ValueError('Build failed')
    return thru.Shape()


# ─── recess shape builders ─────────────────────────────────────────────────────

def _rounded_rect_wire_occ(half_a, half_b, corner_r, center_3d, T, W):
    """Closed OCC wire for a rounded rectangle centred at center_3d."""
    verts_2d = np.array([
        [-half_a, -half_b],
        [ half_a, -half_b],
        [ half_a,  half_b],
        [-half_a,  half_b],
    ], float)
    R = min(float(corner_r), min(half_a, half_b))
    return _flat_panel_wire_occ(verts_2d, R, set(), center_3d, T, W)


def _stamped_recess_occ(rec, n_levels=16):
    """TRUE OCC StampedRecess.

    Lofts n_levels+1 rounded-rectangle wires (parallel offsets of the outer
    footprint at each cross-section level) via ThruSections, then caps the
    bottom with a flat OCC face.  Both shapes are exact OCC B-Rep objects.
    """
    p       = rec.panel
    T       = np.asarray(p.T, float)
    W       = np.asarray(p.W, float)
    Np      = np.asarray(p.N, float)        # outward panel normal
    C3      = np.asarray(p.origin, float) + rec.cx * T + rec.cy * W

    ha      = rec.half_a          # outer footprint half-width
    hb      = rec.half_b          # outer footprint half-height
    R_outer = rec.corner_r
    r_top   = rec.r_top
    r_bot   = rec.r_bot
    wall_h  = rec.wall_h
    α       = rec.draft_angle

    us, vs  = _sample_profile(r_top, wall_h, r_bot, α, n_levels)

    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    added = 0

    # ── wall loft ─────────────────────────────────────────────────────────────
    try:
        thru = BRepOffsetAPI_ThruSections(False, False)   # shell, smooth loft
        wires_added = 0
        for k in range(n_levels + 1):
            u_k = float(us[k])
            v_k = float(vs[k])          # negative → into panel

            # Parallel inward offset of the outer footprint
            a_k = ha - u_k
            b_k = hb - u_k
            R_k = max(0.0, R_outer - u_k)
            if a_k < 1e-6 or b_k < 1e-6:
                continue

            ctr_k  = C3 + v_k * Np     # v_k < 0 → moves into panel (−Np)
            wire_k = _rounded_rect_wire_occ(a_k, b_k, R_k, ctr_k, T, W)
            if wire_k is not None:
                thru.AddWire(wire_k)
                wires_added += 1

        if wires_added >= 2:
            thru.Build()
            if thru.IsDone():
                builder.Add(compound, thru.Shape())
                added += 1
    except Exception:
        pass

    # ── flat bottom face ──────────────────────────────────────────────────────
    try:
        u_bot  = float(us[-1])
        v_bot  = float(vs[-1])
        a_bot  = ha - u_bot
        b_bot  = hb - u_bot
        R_bot  = max(0.0, R_outer - u_bot)
        if a_bot > 1e-6 and b_bot > 1e-6:
            ctr_bot  = C3 + v_bot * Np
            wire_bot = _rounded_rect_wire_occ(a_bot, b_bot, R_bot, ctr_bot, T, W)
            if wire_bot is not None:
                fm = BRepBuilderAPI_MakeFace(wire_bot, True)
                if fm.IsDone():
                    builder.Add(compound, fm.Face())
                    added += 1
    except Exception:
        pass

    if added > 0:
        return compound
    return _compound_from_face_lists(rec.faces())


def _offset_polygon_2d(verts, u):
    """Inward parallel offset of a CCW polygon by distance u.

    Each vertex is moved along the bisector of the two adjacent edge inward
    normals.  Returns (N, 2) array, or None if any offset vertex is invalid.
    """
    n   = len(verts)
    pts = np.asarray(verts, float)
    out = np.empty_like(pts)

    for i in range(n):
        ip = (i - 1) % n
        e_in  = pts[i]  - pts[ip]
        e_out = pts[(i + 1) % n] - pts[i]
        l_in  = np.linalg.norm(e_in)
        l_out = np.linalg.norm(e_out)
        if l_in < 1e-12 or l_out < 1e-12:
            return None
        d_in  = e_in / l_in
        d_out = e_out / l_out
        # Inward (left) perpendiculars for CCW polygon
        n_in  = np.array([-d_in[1],  d_in[0]])
        n_out = np.array([-d_out[1], d_out[0]])
        bisec = n_in + n_out
        blen  = np.linalg.norm(bisec)
        if blen < 1e-9:
            out[i] = pts[i] + u * n_in
        else:
            sin_h = abs(n_in[0] * (bisec / blen)[1] -
                        n_in[1] * (bisec / blen)[0])
            if sin_h < 1e-9:
                out[i] = pts[i] + u * n_in
            else:
                out[i] = pts[i] + (u / sin_h) * (bisec / blen)
    return out


def _polygon_recess_occ(rec, n_levels=16):
    """TRUE OCC PolygonRecess.

    Lofts n_levels+1 polygon-outline wires (inward offset of the outer
    footprint polygon) via ThruSections, then caps with a flat polygon face.
    """
    p       = rec.panel
    T       = np.asarray(p.T, float)
    W       = np.asarray(p.W, float)
    Np      = np.asarray(p.N, float)
    poly_2d = rec.polygon            # (N, 2) outer footprint
    R_outer = rec.corner_r
    r_top   = rec.r_top
    r_bot   = rec.r_bot
    wall_h  = rec.wall_h
    α       = rec.draft_angle

    us, vs  = _sample_profile(r_top, wall_h, r_bot, α, n_levels)
    orig    = np.asarray(p.origin, float)

    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    added = 0

    try:
        thru = BRepOffsetAPI_ThruSections(False, False)
        wires_added = 0

        for k in range(n_levels + 1):
            u_k = float(us[k])
            v_k = float(vs[k])

            poly_k = _offset_polygon_2d(poly_2d, u_k)
            if poly_k is None:
                continue
            R_k      = max(0.0, R_outer - u_k)
            origin_k = orig + v_k * Np
            wire_k   = _flat_panel_wire_occ(poly_k, R_k, set(), origin_k, T, W)
            if wire_k is not None:
                thru.AddWire(wire_k)
                wires_added += 1

        if wires_added >= 2:
            thru.Build()
            if thru.IsDone():
                builder.Add(compound, thru.Shape())
                added += 1
    except Exception:
        pass

    # Flat bottom face
    try:
        u_bot    = float(us[-1])
        v_bot    = float(vs[-1])
        poly_bot = _offset_polygon_2d(poly_2d, u_bot)
        R_bot    = max(0.0, R_outer - u_bot)
        if poly_bot is not None:
            origin_bot = orig + v_bot * Np
            wire_bot   = _flat_panel_wire_occ(poly_bot, R_bot, set(), origin_bot, T, W)
            if wire_bot is not None:
                fm = BRepBuilderAPI_MakeFace(wire_bot, True)
                if fm.IsDone():
                    builder.Add(compound, fm.Face())
                    added += 1
    except Exception:
        pass

    if added > 0:
        return compound
    return _compound_from_face_lists(rec.faces())



# ─── OCC_BracketModel ─────────────────────────────────────────────────────────

class OCC_Renderer(BaseRenderer):
    """Sheet-metal bracket model backed by pythonOCC true B-Rep geometry.

    Drop-in replacement for SheetMetalModel.  All geometry math is delegated
    to the underlying SheetMetalModel; OCC shapes are built lazily at render /
    export time so corner rounding (which depends on registered children) is
    computed correctly.
    """

    def __init__(self, linear_deflection=0.5, angular_deflection=0.5, arc_backend='loft',
                 **kwargs):
        '''
        arc_backend : 'loft'  — arc-guided ThruSections (default, handles any
                                 edge geometry: partial-span, non-parallel,
                                 B-spline curved edges).
                      'cyl'   — exact Geom_CylindricalSurface (BendArc only;
                                 requires aligned, equal-length, straight edges).
        '''
        super().__init__(**kwargs)

        self.arc_backend = arc_backend
        self.linear_deflection = linear_deflection
        self.angular_deflection = angular_deflection

    def tessellate(self, shape):
        """Tessellate an OCC shape into a list of triangles.
        """
    
        mesh = BRepMesh_IncrementalMesh(shape, self.linear_deflection,
                                        False, self.angular_deflection, True)
        mesh.Perform()

        result = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            loc  = TopLoc_Location()
            tri  = BRep_Tool.Triangulation(face, loc)
            if tri and tri.NbTriangles() > 0:
                n_nd = tri.NbNodes()
                pts  = np.array([[tri.Node(i+1).X(),
                                    tri.Node(i+1).Y(),
                                    tri.Node(i+1).Z()]
                                    for i in range(n_nd)])
                for j in range(tri.NbTriangles()):
                    i1, i2, i3 = tri.Triangle(j+1).Get()
                    result.append([pts[i1-1], pts[i2-1], pts[i3-1]])
            exp.Next()
        
        return result


    def render_panel(self, panel: FlatPanel | PolygonPanel | LoftPanel):
        """Single exact OCC face for any panel type.

        FlatPanel: 2-D vertices synthesised from length/width.
        PolygonPanel / SplinePanel: uses panel.vertices directly.
        LoftPanel: ThruSections loft between Curve3D profiles.
        Falls back to mesh compound if construction fails.
        """

        if isinstance(panel, PipePanel): return _pipe_panel_occ(panel)

        elif isinstance(panel, LoftPanel): return _loft_panel_occ(panel)

        elif isinstance(panel, FlatPanel) or isinstance(panel, PolygonPanel):

            verts_2d = np.asarray(panel.vertices, float)
            n        = panel.num_edges
            corner_r = float(panel.corner_r) if hasattr(panel, 'corner_r') else 0.0
            blocked  = {i for i in range(n) if panel._corner_blocked(i)}
            origin   = np.asarray(panel.origin, float)
            T        = np.asarray(panel.T, float)
            W        = np.asarray(panel.W, float)

            wire = _flat_panel_wire_occ(verts_2d, corner_r, blocked, origin, T, W)
            maker = BRepBuilderAPI_MakeFace(wire, True)
            if not maker.IsDone(): raise ValueError('MakeFace failed')
            return maker.Face()

        else:
            raise ValueError(f'Unknown panel type: {type(panel)}')


    def render_arc(self, obj: BendArc | PartialBendArc):
        """Dispatch to the correct OCC arc builder.

        backend : 'loft'  — arc-guided ThruSections loft (default).
                            Handles non-aligned edges, partial spans, B-splines.
                'cyl'   — exact Geom_CylindricalSurface (BendArc only).
                            Correct only for aligned equal-length straight edges.
        """
        if isinstance(obj, PartialBendArc): return _partial_arc_occ(obj)

        # if isinstance(arc, CurvedBendArc):
        #     return _curved_bend_arc_occ(arc)

        elif isinstance(obj, BendArc):
            if self.arc_backend == 'cyl': return _bend_arc_cyl_occ(obj)
            else: return _bend_arc_loft_occ(obj)

        else:
            raise ValueError(f'Unknown arc type: {type(obj)}')


    def render_recess(self, rec: StampedRecess | PolygonRecess):
        """Dispatch to the correct OCC recess builder."""
        if isinstance(rec, StampedRecess): return _stamped_recess_occ(rec)
        elif isinstance(rec, PolygonRecess): return _polygon_recess_occ(rec)
        else: raise ValueError(f'Unknown recess type: {type(rec)}')

    def export_plot(self, groups: dict, markers: dict, ax=None, elev=28., azim=-50.):
        """Export the model to a matplotlib plot.
        """
        tessellated = {}
        for group, shapes in groups.items():
            tessellated[group] = []
            for shape in shapes:
                tessellated[group].append(self.tessellate(shape))

        super().export_plot(tessellated, markers, ax=ax, elev=elev, azim=azim)

    def export_step(self, groups: dict, filepath: str) -> None:
        """Export all shapes to a STEP AP203 file.  Returns True on success.
        """
        writer = STEPControl_Writer()
        for group, shapes in groups.items():
            for shape in shapes:
                writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(filepath)) == IFSelect_RetDone
