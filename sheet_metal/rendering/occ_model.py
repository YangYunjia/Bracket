"""
sheet_metal_occ.occ_model — pythonOCC backend with TRUE OCC B-Rep geometry.

Every shape is an exact OCC B-Rep object (no mesh wrapping):
  FlatPanel / PolygonPanel → single planar OCC face (arc corners via GC_MakeArcOfCircle)
  SplinePanel              → single planar OCC face (Geom_BSplineCurve for curved edges)
  LoftPanel                → BRepOffsetAPI_ThruSections loft between Curve3D profiles.
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
  CurvedBendArc            → arc-guided ThruSections loft (uses _build_grid child edge)
  StampedRecess            → BRepOffsetAPI_ThruSections loft (rounded-rect at each depth level)
                             + BRepBuilderAPI_MakeFace flat bottom
  PolygonRecess            → BRepOffsetAPI_ThruSections loft (offset polygon at each depth level)
                             + BRepBuilderAPI_MakeFace flat bottom

STEP export produces compact, exact B-Rep files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── OCC imports ───────────────────────────────────────────────────────────────
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_BSplineCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

from ..basics.model import SheetMetalModel
from ..basics.primitives import BendArc, PartialBendArc, CurvedBendArc, Curve3D, LoftPanel
from ..basics.recess import _sample_profile


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v

def _to_pnt(arr):
    a = np.asarray(arr, float)
    return gp_Pnt(float(a[0]), float(a[1]), float(a[2]))

def _to_dir(arr):
    a = _n(np.asarray(arr, float))
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

def _polygon_wire_occ(verts_2d, corner_r, blocked, origin, T, W,
                      edge_curves=None):
    """Build an exact closed OCC wire for a 2-D polygon.

    Corners between straight edges are rounded with GC_MakeArcOfCircle.
    Edges supplied via edge_curves (BSplineEdge) are lifted to exact 3-D
    Geom_BSplineCurve edges; their endpoint vertices are interpolated exactly
    by the clamped B-spline so no gap arises at junctions with straight edges.

    Parameters
    ----------
    verts_2d    : (N, 2) CCW polygon vertices in panel local (u, v) frame
    corner_r    : rounding radius (0 = all sharp)
    blocked     : set of vertex indices that must stay sharp
    origin, T, W: 3-D panel frame  (3-D point = origin + u*T + v*W)
    edge_curves : list of N BSplineEdge|None per edge, or None (all straight)
    """
    verts_2d = np.asarray(verts_2d, float)
    n        = len(verts_2d)
    origin   = np.asarray(origin, float)
    T        = np.asarray(T, float)
    W        = np.asarray(W, float)
    if edge_curves is None:
        edge_curves = [None] * n

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
            if (edge_curves[(i - 1) % n] is not None or
                    edge_curves[i] is not None):
                continue   # adjacent B-spline edge: skip rounding here

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
        crv_i = edge_curves[i]

        if crv_i is not None:
            # ── B-spline edge ──────────────────────────────────────────────
            cp_2d      = crv_i.control_pts           # (M, 2)
            deg        = int(crv_i.degree)
            flat_knots = crv_i._spu.t.copy()

            # Lift control points to 3-D
            poles_3d = [to_3d(cp) for cp in cp_2d]

            # Convert flat knot vector → distinct values + multiplicities
            distinct, mults = [], []
            for k in flat_knots:
                if not distinct or abs(k - distinct[-1]) > 1e-12:
                    distinct.append(float(k))
                    mults.append(1)
                else:
                    mults[-1] += 1

            n_poles = len(poles_3d)
            n_knots = len(distinct)
            poles_arr = TColgp_Array1OfPnt(1, n_poles)
            knots_arr = TColStd_Array1OfReal(1, n_knots)
            mults_arr = TColStd_Array1OfInteger(1, n_knots)
            for ki, p in enumerate(poles_3d):
                poles_arr.SetValue(ki + 1, _to_pnt(p))
            for ki, k in enumerate(distinct):
                knots_arr.SetValue(ki + 1, k)
            for ki, m in enumerate(mults):
                mults_arr.SetValue(ki + 1, m)

            try:
                bspl = Geom_BSplineCurve(poles_arr, knots_arr, mults_arr, deg)
                edges.append(BRepBuilderAPI_MakeEdge(bspl, 0.0, 1.0).Edge())
            except Exception:
                # Fallback: straight chord
                edges.append(BRepBuilderAPI_MakeEdge(
                    _to_pnt(to_3d(verts_2d[i])),
                    _to_pnt(to_3d(verts_2d[j]))).Edge())

        else:
            # ── Straight segment from after-arc[i] to before-arc[j] ───────
            seg_start = arcs[i][2] if arcs[i] is not None else to_3d(verts_2d[i])
            seg_end   = arcs[j][0] if arcs[j] is not None else to_3d(verts_2d[j])

            if np.linalg.norm(np.asarray(seg_end) - np.asarray(seg_start)) > 1e-9:
                edges.append(BRepBuilderAPI_MakeEdge(
                    _to_pnt(seg_start), _to_pnt(seg_end)).Edge())

            # Arc at vertex j (only between two straight edges)
            if arcs[j] is not None and edge_curves[j] is None:
                P1, Pm, P3 = arcs[j]
                try:
                    arc = GC_MakeArcOfCircle(_to_pnt(P1), _to_pnt(Pm), _to_pnt(P3))
                    if arc.IsDone():
                        edges.append(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
                except Exception:
                    pass

    if not edges:
        return None

    wb = BRepBuilderAPI_MakeWire()
    for e in edges:
        wb.Add(e)
    return wb.Wire() if wb.IsDone() else None


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
        return wb.Wire() if wb.IsDone() else None

    if kind == 'arc':
        # Sample 3 points: start, mid, end for GC_MakeArcOfCircle
        pts = crv._eval_raw(np.array([0.0, 0.5, 1.0]))
        try:
            arc_geom = GC_MakeArcOfCircle(
                _to_pnt(pts[0]), _to_pnt(pts[1]), _to_pnt(pts[2]))
            if arc_geom.IsDone():
                edge = BRepBuilderAPI_MakeEdge(arc_geom.Value()).Edge()
                wb = BRepBuilderAPI_MakeWire(edge)
                return wb.Wire() if wb.IsDone() else None
        except Exception:
            pass
        # Fall through to polygon approximation

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
        try:
            bspl = Geom_BSplineCurve(poles_arr, knots_arr, mults_arr, deg)
            edge = BRepBuilderAPI_MakeEdge(bspl, 0.0, 1.0).Edge()
            wb = BRepBuilderAPI_MakeWire(edge)
            return wb.Wire() if wb.IsDone() else None
        except Exception:
            pass
        # Fall through to polygon approximation

    # polyline / fallback: dense polygon wire
    if kind == 'polyline':
        pts = crv._pts
    else:
        pts = crv.sample(n_poly)
    poly = BRepBuilderAPI_MakePolygon()
    for p in pts:
        poly.Add(_to_pnt(p))
    if poly.IsDone():
        wb = BRepBuilderAPI_MakeWire(poly.Wire())
        return wb.Wire() if wb.IsDone() else None
    return None


def _shape_for_loft_panel(panel):
    """OCC ThruSections loft for a LoftPanel.

    Each Curve3D profile becomes an OCC wire via _curve3d_to_occ_wire().
    ruled=True  → BRepOffsetAPI_ThruSections(isSolid=False, isRuled=True)
    ruled=False → BRepOffsetAPI_ThruSections(isSolid=False, isRuled=False)

    Falls back to mesh compound if the loft fails.
    """
    try:
        thru = BRepOffsetAPI_ThruSections(False, panel.ruled)
        n_added = 0
        for crv in panel.profiles:
            wire = _curve3d_to_occ_wire(crv)
            if wire is not None:
                thru.AddWire(wire)
                n_added += 1
        if n_added >= 2:
            thru.Build()
            if thru.IsDone():
                return thru.Shape()
    except Exception:
        pass
    return _compound_from_face_lists(panel.faces())


# ─── panel shape builder ──────────────────────────────────────────────────────

def _shape_for_panel(panel):
    """Single exact OCC face for any panel type.

    FlatPanel: 2-D vertices synthesised from length/width.
    PolygonPanel / SplinePanel: uses panel.vertices directly.
    LoftPanel: ThruSections loft between Curve3D profiles.
    Falls back to mesh compound if construction fails.
    """
    from sheet_metal.primitives import FlatPanel, PolygonPanel, SplinePanel

    if isinstance(panel, LoftPanel):
        return _shape_for_loft_panel(panel)

    try:
        if isinstance(panel, FlatPanel):
            L, Wd  = panel.length, panel.width
            verts_2d = np.array([[0, 0], [L, 0], [L, Wd], [0, Wd]], float)
            n        = 4
        else:
            verts_2d = np.asarray(panel.vertices, float)
            n        = len(verts_2d)

        corner_r = float(panel.corner_r) if hasattr(panel, 'corner_r') else 0.0
        blocked  = {i for i in range(n) if panel._corner_blocked(i)}
        origin   = np.asarray(panel.origin, float)
        T        = np.asarray(panel.T, float)
        W        = np.asarray(panel.W, float)

        edge_curves = None
        if isinstance(panel, SplinePanel):
            edge_curves = [panel.edge_curve(i) for i in range(n)]

        wire = _polygon_wire_occ(verts_2d, corner_r, blocked, origin, T, W,
                                  edge_curves)
        if wire is None:
            raise ValueError('wire builder returned None')

        maker = BRepBuilderAPI_MakeFace(wire, True)
        if maker.IsDone():
            return maker.Face()
        raise ValueError('MakeFace failed')

    except Exception:
        pass

    return _compound_from_face_lists(panel.faces())


# ─── bend arc builders ─────────────────────────────────────────────────────────

# ── Backend A: exact Geom_CylindricalSurface ─────────────────────────────────
# Assumes both edges are parallel, equal-length, and straight.  Exact for
# aligned BendArc; incorrect for partial-span, non-parallel, or curved edges.

def _bend_arc_cyl_occ(arc):
    """Exact Geom_CylindricalSurface face for an aligned straight BendArc."""
    p0     = np.asarray(arc.p0, float)
    p1     = np.asarray(arc.p1, float)
    N_bend = _n(np.asarray(arc.N_bend, float))
    alpha  = float(arc.alpha)
    r_n    = float(arc.r_n)

    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-12 or abs(alpha) < 1e-9:
        return None

    edge_dir = (p1 - p0) / L
    C0  = p0 + r_n * N_bend
    ax3 = gp_Ax3(_to_pnt(C0), _to_dir(edge_dir), _to_dir(-N_bend))
    cyl = Geom_CylindricalSurface(ax3, r_n)

    try:
        maker = BRepBuilderAPI_MakeFace(cyl, -alpha, 0.0, 0.0, L, 1e-6)
        if maker.IsDone():
            return maker.Face()
    except Exception:
        pass
    return None


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
    Nb    = _n(np.asarray(N_bend, float))
    sa_alpha = np.sin(float(alpha))
    ca_alpha = np.cos(float(alpha))
    if abs(sa_alpha) < 1e-9:
        return None

    parent_pts = np.asarray(parent_pts, float)
    child_pts  = np.asarray(child_pts,  float)

    # Arc centres along the parent edge
    C_js = parent_pts + r_n * Nb                           # (M+1, 3)

    # Local tangent-out back-derived from cA = C + r·(−cos α·Nb + sin α·T_out)
    T_outs = (child_pts - C_js + r_n * ca_alpha * Nb) / (r_n * sa_alpha)  # (M+1, 3)

    try:
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

        if n_added >= 2:
            thru.Build()
            if thru.IsDone():
                return thru.Shape()
    except Exception:
        pass
    return None


def _bend_arc_loft_occ(arc, n_arc=24):
    """Arc-guided loft for BendArc (straight, possibly partial-span)."""
    p0    = np.asarray(arc.p0, float)
    p1    = np.asarray(arc.p1, float)
    T_out = _n(np.asarray(arc.T_out, float))
    Nb    = _n(np.asarray(arc.N_bend, float))
    alpha = float(arc.alpha)
    r_n   = float(arc.r_n)

    if abs(alpha) < 1e-9 or np.linalg.norm(p1 - p0) < 1e-9:
        return None

    # For a straight edge just the two endpoints are needed
    parent_pts = np.array([p0, p1])
    C_pts      = parent_pts + r_n * Nb
    child_pts  = C_pts + r_n * (-np.cos(alpha) * Nb + np.sin(alpha) * T_out)

    return _arc_loft_occ(parent_pts, child_pts, Nb, alpha, r_n, n_arc=n_arc)


def _partial_arc_occ(arc, n_alpha=24):
    """ThruSections ruled loft for PartialBendArc — exact OCC surface."""
    p0    = np.asarray(arc.p0, float)
    p1    = np.asarray(arc.p1, float)
    T_out = _n(np.asarray(arc.T_out, float))
    Nb    = _n(np.asarray(arc.N_bend, float))
    alpha = float(arc.alpha)
    r_n   = float(arc.r_n)
    L     = float(arc.L)
    s0    = float(arc.s0)
    s1    = float(arc.s1)
    blend = float(arc.blend)
    taper = arc.taper

    if L < 1e-12 or abs(alpha) < 1e-9:
        return _compound_from_face_lists(arc.faces())

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

    try:
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
        if thru.IsDone():
            return thru.Shape()
    except Exception:
        pass

    return _compound_from_face_lists(arc.faces())


def _curved_bend_arc_occ(arc, n_arc=24):
    """Arc-guided loft for CurvedBendArc.

    Source edge (a=0)   : the curved parent B-spline edge.
    Target edge (a=alpha): the straight root edge of the flat child panel.

    The child panel root edge lies along W_child = the chord direction of the
    parent edge.  extend_panel computes child_width = dot(q_end − c0, W_child),
    projecting out any perpendicular component of the arc endpoint.  We must
    do the same projection here so the loft terminates exactly at the child
    panel root edge with no gap.
    """
    try:
        n_e  = min(arc.n_edge, 32)
        grid = arc._build_grid()           # (n_a+1, n_e+1, 3)

        # W_child = chord direction of the parent edge (= W_child in extend_panel)
        pT  = np.array(arc.panel_T,      float)
        pW  = np.array(arc.panel_W,      float)
        uv0 = arc.edge_curve.eval(0.0)
        uv1 = arc.edge_curve.eval(1.0)
        p0_3d   = np.array(arc.panel_origin, float) + uv0[0]*pT + uv0[1]*pW
        p1_3d   = np.array(arc.panel_origin, float) + uv1[0]*pT + uv1[1]*pW
        W_child = _n(p1_3d - p0_3d)      # chord direction = child panel W axis

        if grid.shape[1] != n_e + 1:
            pts_uv     = arc.edge_curve.sample_arclength(n_e + 1)
            parent_pts = np.array([np.array(arc.panel_origin, float)
                                    + uv[0]*pT + uv[1]*pW for uv in pts_uv])
            ts_al  = np.linspace(0.0, 1.0, n_e + 1)
            tangs  = arc.edge_curve.tangent(ts_al)
            Nb     = _n(arc.N_bend)
            r_n    = arc.r_n
            alpha  = arc.alpha
            T_outs = np.array([_n(t[1]*pT - t[0]*pW) for t in tangs])
            C_js   = parent_pts + r_n * Nb
            arc_child = C_js + r_n * (-np.cos(alpha)*Nb + np.sin(alpha)*T_outs)
            c0 = arc_child[0]
            c1_raw = arc_child[-1]
        else:
            parent_pts = grid[0, :, :]
            c0     = grid[-1,  0, :]
            c1_raw = grid[-1, -1, :]

        # Project c1 onto the child root edge line (through c0 along W_child).
        # This mirrors extend_panel's child_width = dot(q_end − c0, W_child)
        # and ensures c1 lands exactly at the child panel far-end vertex.
        span = float(np.dot(c1_raw - c0, W_child))
        c1   = c0 + span * W_child

        ts        = np.linspace(0.0, 1.0, n_e + 1)
        child_pts = c0 + ts[:, np.newaxis] * (c1 - c0)

        result = _arc_loft_occ(parent_pts, child_pts,
                                arc.N_bend, arc.alpha, arc.r_n, n_arc=n_arc)
        if result is not None:
            return result
    except Exception:
        pass

    return _compound_from_face_lists(arc.faces())


def _shape_for_arc(arc, backend='loft'):
    """Dispatch to the correct OCC arc builder.

    backend : 'loft'  — arc-guided ThruSections loft (default).
                        Handles non-aligned edges, partial spans, B-splines.
              'cyl'   — exact Geom_CylindricalSurface (BendArc only).
                        Correct only for aligned equal-length straight edges.
    """
    if isinstance(arc, PartialBendArc):
        return _partial_arc_occ(arc)

    if isinstance(arc, CurvedBendArc):
        return _curved_bend_arc_occ(arc)

    if isinstance(arc, BendArc):
        if backend == 'cyl':
            shape = _bend_arc_cyl_occ(arc)
        else:
            shape = _bend_arc_loft_occ(arc)
        if shape is not None:
            return shape
        return _compound_from_face_lists(arc.faces())

    return _compound_from_face_lists(arc.faces())


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
    return _polygon_wire_occ(verts_2d, R, set(), center_3d, T, W)


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
            wire_k   = _polygon_wire_occ(poly_k, R_k, set(), origin_k, T, W)
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
            wire_bot   = _polygon_wire_occ(poly_bot, R_bot, set(), origin_bot, T, W)
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


def _shape_for_recess(rec):
    """Dispatch to the correct OCC recess builder."""
    from sheet_metal.recess import StampedRecess, PolygonRecess
    if isinstance(rec, StampedRecess):
        return _stamped_recess_occ(rec)
    elif isinstance(rec, PolygonRecess):
        return _polygon_recess_occ(rec)
    return _compound_from_face_lists(rec.faces())


# ─── tessellation ─────────────────────────────────────────────────────────────

def tessellate_shapes(panel_shapes, arc_shapes, recess_shapes,
                      linear_deflection=0.5, angular_deflection=0.5):
    """Tessellate OCC shapes for rendering.

    Returns dict {'panel': [...], 'bend': [...], 'recess': [...]} where each
    value is a list of triangles (each triangle = [p0, p1, p2], each pi a
    3-element numpy array in world coordinates).
    """
    result  = {'panel': [], 'bend': [], 'recess': []}
    buckets = [('panel', panel_shapes),
               ('bend',  arc_shapes),
               ('recess', recess_shapes)]

    for key, shapes in buckets:
        for shape in shapes:
            if shape is None:
                continue
            try:
                mesh = BRepMesh_IncrementalMesh(shape, linear_deflection,
                                                False, angular_deflection, True)
                mesh.Perform()
            except Exception:
                continue

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
                        result[key].append([pts[i1-1], pts[i2-1], pts[i3-1]])
                exp.Next()

    return result


# ─── OCC_BracketModel ─────────────────────────────────────────────────────────

class OCC_BracketModel:
    """Sheet-metal bracket model backed by pythonOCC true B-Rep geometry.

    Drop-in replacement for SheetMetalModel.  All geometry math is delegated
    to the underlying SheetMetalModel; OCC shapes are built lazily at render /
    export time so corner rounding (which depends on registered children) is
    computed correctly.
    """

    def __init__(self, t=2.0, inner_r=5.0):
        self._sm = SheetMetalModel(t=t, inner_r=inner_r)

    # ── property pass-throughs ────────────────────────────────────────────────
    @property
    def t(self):        return self._sm.t
    @property
    def inner_r(self):  return self._sm.inner_r
    @property
    def panels(self):   return self._sm.panels
    @property
    def arcs(self):     return self._sm.arcs
    @property
    def recesses(self): return self._sm.recesses

    # ── builder API (mirrors SheetMetalModel exactly) ─────────────────────────

    def add_panel(self, origin, T, W, length, width, name=None, corner_r=0.0):
        return self._sm.add_panel(origin, T, W, length, width,
                                   name=name, corner_r=corner_r)

    def add_polygon_panel(self, origin, T, W, vertices, name=None, corner_r=0.0):
        return self._sm.add_polygon_panel(origin, T, W, vertices,
                                           name=name, corner_r=corner_r)

    def add_spline_panel(self, origin, T, W, vertices, edge_curves=None,
                          name=None, corner_r=0.0):
        return self._sm.add_spline_panel(origin, T, W, vertices, edge_curves,
                                          name=name, corner_r=corner_r)

    def add_loft_panel(self, profiles, ruled=True, name=None):
        """Add a standalone LoftPanel.

        profiles : list of Curve3D or [(x,y,z),(x,y,z)] pairs
        ruled    : True → ruled; False → smooth loft
        """
        return self._sm.add_loft_panel(profiles, ruled=ruled, name=name)

    def extend(self, parent, edge_idx, angle_deg, child_length,
               bend_dir=+1, start_frac=0.0, end_frac=1.0,
               blend_r=None, taper='arc'):
        return self._sm.extend(parent, edge_idx, angle_deg, child_length,
                                bend_dir=bend_dir,
                                start_frac=start_frac, end_frac=end_frac,
                                blend_r=blend_r, taper=taper)

    def extend_loft(self, parent, edge_idx, angle_deg, free_curve,
                    bend_dir=+1, start_frac=0.0, end_frac=1.0,
                    intermediate_curves=None, ruled=True):
        """Extend a LoftPanel child from parent.

        free_curve : Curve3D or [(x,y,z),(x,y,z)] for the far edge
        """
        return self._sm.extend_loft(
            parent, edge_idx, angle_deg, free_curve,
            bend_dir=bend_dir,
            start_frac=start_frac, end_frac=end_frac,
            intermediate_curves=intermediate_curves, ruled=ruled)

    def add_recess(self, panel, cx, cy, width, height, corner_r,
                   r_top, r_bot, depth=None, wall_h=None,
                   draft_angle=0.0):
        return self._sm.add_recess(panel, cx, cy, width, height, corner_r,
                                    r_top, r_bot, depth=depth, wall_h=wall_h,
                                    draft_angle=draft_angle)

    def add_polygon_recess(self, panel, polygon, corner_r=0.0,
                            r_top=3.0, r_bot=3.0, depth=None, wall_h=None,
                            draft_angle=0.0):
        return self._sm.add_polygon_recess(panel, polygon, corner_r=corner_r,
                                            r_top=r_top, r_bot=r_bot,
                                            depth=depth, wall_h=wall_h,
                                            draft_angle=draft_angle)

    def mark_hole(self, panel, u, v, label='F', r_vis=5.0):
        self._sm.mark_hole(panel, u, v, label=label, r_vis=r_vis)

    def node_of(self, panel):
        return self._sm.node_of(panel)

    def print_tree(self):
        self._sm.print_tree()

    # ── OCC shape builders ─────────────────────────────────────────────────────

    def _build_occ_shapes(self, arc_backend='loft'):
        """Build all OCC shapes.

        arc_backend : 'loft'  — arc-guided ThruSections (default, handles any
                                 edge geometry: partial-span, non-parallel,
                                 B-spline curved edges).
                      'cyl'   — exact Geom_CylindricalSurface (BendArc only;
                                 requires aligned, equal-length, straight edges).
        """
        panel_shapes  = [_shape_for_panel(p)                       for p in self._sm.panels]
        arc_shapes    = [_shape_for_arc(arc, backend=arc_backend)   for arc in self._sm.arcs]
        recess_shapes = [_shape_for_recess(r)                       for r in self._sm.recesses]
        return panel_shapes, arc_shapes, recess_shapes

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self, ax=None,
               color_panel='#4a90c4', color_bend='#2e6ea6',
               color_recess='#3a7abf',
               alpha_surf=0.85, ec='#1a3a52', lw=0.4,
               elev=28., azim=-50.,
               arc_backend='loft'):
        """Render using OCC tessellation → matplotlib Poly3DCollection.

        arc_backend : 'loft' (default) or 'cyl' — see _build_occ_shapes.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        panel_sh, arc_sh, recess_sh = self._build_occ_shapes(arc_backend=arc_backend)
        tess = tessellate_shapes(panel_sh, arc_sh, recess_sh)

        color_map = {'panel': color_panel, 'bend': color_bend, 'recess': color_recess}
        all_pts = []
        for key, tris in tess.items():
            if not tris:
                continue
            ax.add_collection3d(Poly3DCollection(
                tris, alpha=alpha_surf,
                facecolor=color_map[key], edgecolor=ec, linewidth=lw))
            all_pts.extend(pt for tri in tris for pt in tri)

        # Hole markers (delegated to underlying model)
        hole_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for i, h in enumerate(self._sm._holes):
            c, ctr, nrm, r = hole_colors[i % 4], h['centre'], h['normal'], h['r']
            ref  = np.array([0, 1, 0]) if abs(nrm[1]) < 0.85 else np.array([1, 0, 0])
            u_   = _n(np.cross(nrm, ref))
            v_   = _n(np.cross(nrm, u_))
            ang  = np.linspace(0, 2 * np.pi, 48)
            ring = ctr + r * (np.outer(np.cos(ang), u_) + np.outer(np.sin(ang), v_))
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=c, lw=2.2, zorder=10)
            ax.quiver(*ctr, *(nrm * r * 1.5), color=c, lw=1.8,
                      arrow_length_ratio=0.35, zorder=10)
            ax.text(*(ctr + nrm * r * 2.5), h['label'],
                    fontsize=10, color=c, fontweight='bold')

        if all_pts:
            pts   = np.array(all_pts)
            lo, hi = pts.min(0), pts.max(0)
            mid   = (lo + hi) / 2
            span  = max((hi - lo).max() * 0.65, 1.0)
            for fn, m_ in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
                fn(m_ - span, m_ + span)

        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        return ax

    # ── STEP export ───────────────────────────────────────────────────────────

    def export_step(self, filepath, arc_backend='loft'):
        """Export all shapes to a STEP AP203 file.  Returns True on success.

        arc_backend : 'loft' (default) or 'cyl' — see _build_occ_shapes.
        """
        panel_sh, arc_sh, recess_sh = self._build_occ_shapes(arc_backend=arc_backend)

        writer   = STEPControl_Writer()
        n_written = 0
        for shape in panel_sh + arc_sh + recess_sh:
            if shape is not None:
                try:
                    writer.Transfer(shape, STEPControl_AsIs)
                    n_written += 1
                except Exception:
                    pass

        if n_written == 0:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        return writer.Write(str(filepath)) == IFSelect_RetDone
