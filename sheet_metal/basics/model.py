"""
sheet_metal.model — SheetMetalModel builder and renderer.
"""

import numpy as np

from .maths import n
from .panel import FlatPanel, PolygonPanel, LoftPanel, Curve3D
from .bending import BendArc, PartialBendArc
from .tree import PanelNode
from .recess import StampedRecess, PolygonRecess

'''
def _arc_transform_curve(parent_crv, parent, alpha, r_n, bdir,
                         T_child, W_child, child_origin):
    """Return a BSplineEdge for the child's root edge (edge index 3).

    Transforms each control point of the parent edge curve through the arc
    geometry to produce the corresponding child-side curve in the child's
    local (T_child, W_child) frame.  The result is reversed so that eval(0)
    corresponds to the C3 endpoint and eval(1) to C0 (matching edge-3
    direction in the CCW child SplinePanel).
    """
    cp_2d = parent_crv.control_pts   # (M, 2) in parent (u, v)
    M     = len(cp_2d)
    ts    = np.linspace(0.0, 1.0, M)

    child_v = []
    for ti, (ui, vi) in zip(ts, cp_2d):
        p_i    = parent.origin + ui*parent.T + vi*parent.W
        tang   = parent_crv.tangent(float(ti))
        tn     = np.linalg.norm(tang)
        if tn < 1e-12:
            tang = np.array([1.0, 0.0])
            tn   = 1.0
        tang      /= tn
        T_out_i    = n(tang[1]*parent.T - tang[0]*parent.W)
        C_i        = p_i + r_n * bdir
        q_i        = C_i + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out_i)
        child_v.append(float(np.dot(q_i - child_origin, W_child)))

    # Edge 3 in the child CCW polygon goes from C3 (v=max) to C0 (v=0).
    # The arc-transformed curve goes C0→C3 (v increases), so reverse it.
    child_cp = [[0.0, v] for v in reversed(child_v)]
    return BSplineEdge(child_cp, parent_crv.degree)
'''

def extend_panel(parent, edge_idx, angle_deg, child_length,
                 inner_r=5.0, bend_dir=+1,
                 start_frac=0.0, end_frac=1.0,
                 blend_r=None, taper='arc'):
    """Extend a child panel from one edge of a parent panel.

    Parameters
    ----------
    parent      : FlatPanel | PolygonPanel | SplinePanel
    edge_idx    : edge index on the parent panel
    angle_deg   : bend angle in degrees
    child_length: panel length after the bend
    inner_r     : inner bend radius (mm)
    bend_dir    : +1 toward +N_parent, -1 toward -N_parent
    start_frac  : fraction along edge where child starts  (0–1)
    end_frac    : fraction along edge where child ends    (0–1)
    blend_r     : lateral blend half-width for partial bends (default = r_n)
    taper       : 'arc' or 'linear' (PartialBendArc only)

    Returns
    -------
    (child, arc)
    For straight parent edges : (FlatPanel,    BendArc | PartialBendArc)
    For curved  parent edges  : (SplinePanel,  CurvedBendArc)
                                (partial curved edges use PartialBendArc as approx)
    """
    p0_full, p1_full, edge_dir, T_out = parent.edge(edge_idx)
    L_edge  = float(np.linalg.norm(p1_full - p0_full))
    ed_unit = (p1_full - p0_full) / L_edge

    # Trim the reference edge to the flat (non-rounded-corner) region.
    if hasattr(parent, '_edge_flat_trims'):
        ts, te = parent._edge_flat_trims(edge_idx)
        if start_frac > 1e-6:
            p0_full = p0_full + ts * ed_unit
        if end_frac < 1.0 - 1e-6:
            p1_full = p1_full - te * ed_unit
        L_edge = float(np.linalg.norm(p1_full - p0_full))

    p0 = p0_full + start_frac * (p1_full - p0_full)
    p1 = p0_full + end_frac   * (p1_full - p0_full)

    t     = parent.t
    alpha = np.radians(abs(angle_deg))
    # Use edge-local normal when available (e.g. LoftPanel has varying normal).
    N_src = (parent.N_at_edge(edge_idx)
             if hasattr(parent, 'N_at_edge')
             else parent.N)
    bdir  = float(bend_dir) * N_src
    r_n   = inner_r + t / 2
    # Check for a curved parent edge (SplinePanel only).
    crv = parent.edge_curve(edge_idx) if hasattr(parent, 'edge_curve') else None

    '''
    if crv is not None and not ((start_frac > 1e-6) or (end_frac < 1.0 - 1e-6)):
        # ── Curved full-edge path ──────────────────────────────────────────────
        # T_out at the start of the child's span from the curve tangent.
        tang_start  = crv.tangent(float(start_frac))
        tn          = np.linalg.norm(tang_start)
        if tn > 1e-12:
            tang_start /= tn
        T_out_start = n(tang_start[1]*parent.T - tang_start[0]*parent.W)

        T_child = n(np.sin(alpha)*bdir + np.cos(alpha)*T_out_start)
        W_child = edge_dir   # chord direction — child is flat

        C0           = p0 + r_n * bdir
        child_origin = C0 + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out_start)

        # child_width = projected distance from C0 to C3 along W_child.
        tang_end = crv.tangent(float(end_frac))
        tn_e     = np.linalg.norm(tang_end)
        if tn_e > 1e-12:
            tang_end /= tn_e
        T_out_end  = n(tang_end[1]*parent.T - tang_end[0]*parent.W)
        C_end      = p1 + r_n * bdir
        q_end      = C_end + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out_end)
        child_width = float(np.dot(q_end - child_origin, W_child))
        if child_width < 1e-6:
            child_width = float(np.linalg.norm(p1 - p0))

        # Build child SplinePanel with arc-transformed root edge (edge 3).
        root_crv = _arc_transform_curve(
            crv, parent, alpha, r_n, bdir, T_child, W_child, child_origin)
        child = SplinePanel(
            origin=child_origin, T=T_child, W=W_child,
            vertices=[[0, 0], [child_length, 0],
                      [child_length, child_width], [0, child_width]],
            edge_curves=[None, None, None, root_crv],
            t=t, name=f'{parent.name}.e{edge_idx}',
        )

        arc = CurvedBendArc(
            edge_curve=crv,
            panel_origin=parent.origin, panel_T=parent.T, panel_W=parent.W,
            N_bend=bdir, alpha=alpha, r_n=r_n,
        )
        return child, arc
    '''

    # ── Straight-edge path (or partial curved — uses chord approximation) ──────
    T_child     = n(np.sin(alpha)*bdir + np.cos(alpha)*T_out)
    W_child     = edge_dir
    child_width = float(np.linalg.norm(p1 - p0))

    C0           = p0 + r_n * bdir
    child_origin = C0 + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out)

    child = FlatPanel(
        origin=child_origin, T=T_child, W=W_child,
        length=child_length, width=child_width,
        t=t, name=f'{parent.name}.e{edge_idx}',
    )

    is_partial = (start_frac > 1e-6) or (end_frac < 1.0 - 1e-6)

    if is_partial:
        _br = float(blend_r) if blend_r is not None else r_n
        arc = PartialBendArc(
            p0_full=p0_full, p1_full=p1_full,
            s_start=start_frac * L_edge,
            s_end=end_frac     * L_edge,
            T_out=T_out, N_bend=bdir,
            alpha=alpha, r_n=r_n,
            blend_r=_br, taper=taper,
        )
    else:
        arc = BendArc(p0=p0, p1=p1, T_out=T_out, N_bend=bdir,
                      alpha=alpha, r_n=r_n)

    return child, arc


def extend_loft(parent, edge_idx, angle_deg, free_curve,
                inner_r=5.0, bend_dir=+1,
                start_frac=0.0, end_frac=1.0,
                intermediate_curves=None, ruled=True):
    """Extend a LoftPanel child from a parent panel edge.

    Works like extend_panel() but creates a LoftPanel whose free edge is
    the user-supplied Curve3D `free_curve` instead of a flat FlatPanel.

    Parameters
    ----------
    parent             : FlatPanel | PolygonPanel | SplinePanel | LoftPanel
    edge_idx           : edge index on the parent
    angle_deg          : bend angle in degrees
    free_curve         : Curve3D (or [(x,y,z),(x,y,z)] shorthand) for the
                         free (far) edge of the child panel
    inner_r            : inner bend radius
    bend_dir           : +1 toward +N_parent, -1 toward -N_parent
    start_frac         : fraction along parent edge where child starts (0–1)
    end_frac           : fraction along parent edge where child ends   (0–1)
    intermediate_curves: optional list of Curve3D for intermediate profiles
    ruled              : True → ruled surface; False → smooth loft

    Returns
    -------
    (child_LoftPanel, arc)
    """
    # ── Compute root edge geometry (same logic as extend_panel) ───────────────
    p0_full, p1_full, edge_dir, T_out = parent.edge(edge_idx)
    L_edge  = float(np.linalg.norm(p1_full - p0_full))
    ed_unit = (p1_full - p0_full) / L_edge

    if hasattr(parent, '_edge_flat_trims'):
        ts, te = parent._edge_flat_trims(edge_idx)
        if start_frac > 1e-6:
            p0_full = p0_full + ts * ed_unit
        if end_frac < 1.0 - 1e-6:
            p1_full = p1_full - te * ed_unit
        L_edge = float(np.linalg.norm(p1_full - p0_full))

    p0 = p0_full + start_frac * (p1_full - p0_full)
    p1 = p0_full + end_frac   * (p1_full - p0_full)

    t     = parent.t
    alpha = np.radians(abs(angle_deg))
    N_src = (parent.N_at_edge(edge_idx)
             if hasattr(parent, 'N_at_edge') else parent.N)
    bdir  = float(bend_dir) * N_src
    r_n   = inner_r + t / 2
    # Root edge of the child: straight line from p0_child to p1_child
    T_child = n(np.sin(alpha)*bdir + np.cos(alpha)*T_out)
    C0           = p0 + r_n * bdir
    child_origin = C0 + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out)
    C1           = p1 + r_n * bdir
    child_far    = C1 + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out)

    root_curve = Curve3D.line(child_origin, child_far)

    # Coerce free_curve if a raw point list was passed
    if not isinstance(free_curve, Curve3D):
        pts = np.asarray(free_curve, float)
        if pts.ndim == 2 and len(pts) == 2:
            free_curve = Curve3D.line(pts[0], pts[1])
        else:
            free_curve = Curve3D('polyline', _pts=pts)

    profiles = [root_curve]
    if intermediate_curves:
        profiles += list(intermediate_curves)
    profiles.append(free_curve)

    child = LoftPanel(profiles=profiles, ruled=ruled,
                      name=f'{parent.name}.e{edge_idx}', t=t)

    # Bend arc: always a straight BendArc along the straight root edge
    is_partial = (start_frac > 1e-6) or (end_frac < 1.0 - 1e-6)
    if is_partial:
        arc = PartialBendArc(
            p0_full=p0_full, p1_full=p1_full,
            s_start=start_frac * L_edge,
            s_end=end_frac     * L_edge,
            T_out=T_out, N_bend=bdir,
            alpha=alpha, r_n=r_n,
        )
    else:
        arc = BendArc(p0=p0, p1=p1, T_out=T_out, N_bend=bdir,
                      alpha=alpha, r_n=r_n)

    return child, arc


class SheetMetalModel:
    """Incrementally built sheet-metal mid-surface model.

    Panel connectivity is recorded as a forest of PanelNode trees — one root
    per add_panel() call.  Any panel (including ones produced by extend()) can
    be used as the parent for further extend() calls.

    Tree queries
    ------------
    m.node_of(panel)   → PanelNode
    m.print_tree()     → ASCII tree of the full structure
    """

    def __init__(self, t=2.0, inner_r=5.0):
        self.t         = float(t)
        self.inner_r   = float(inner_r)
        self.panels    = []        # all FlatPanel objects in insertion order
        self.arcs      = []        # all arc objects in insertion order
        self.recesses  = []        # all StampedRecess objects
        self._holes    = []
        self.roots     = []        # root PanelNodes (one per add_panel call)
        self._node_map = {}        # id(FlatPanel) → PanelNode

    def add_panel(self, origin, T, W, length, width, name=None, corner_r=0.0):
        if name is None:
            name = f'P{len(self.panels)}'
        p = FlatPanel(origin, T, W, length, width, self.t, name, corner_r=corner_r)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def add_spline_panel(self, origin, T, W, vertices, edge_curves=None,
                         name=None, corner_r=0.0):
        """Add a flat panel whose edges can be B-spline curves.

        origin      : 3-D position of the panel coordinate origin
        T, W        : two orthogonal 3-D vectors spanning the panel plane
        vertices    : (N, 2) list/array of (u, v) corners in the T-W local frame
        edge_curves : list of N items (BSplineEdge | None); None = straight edge
        name        : optional label
        corner_r    : rounding radius at free straight-edge corners

        Returns the SplinePanel.
        """
        if name is None:
            name = f'P{len(self.panels)}'
        p = SplinePanel(origin, T, W, vertices, edge_curves,
                        self.t, name, corner_r=corner_r)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def add_polygon_panel(self, origin, T, W, vertices, name=None, corner_r=0.0):
        """Add a flat panel with an arbitrary polygon outline.

        origin   : 3-D position of the panel coordinate origin
        T, W     : two orthogonal 3-D vectors spanning the panel plane
        vertices : (N, 2) list/array of (u, v) pairs in the T-W local frame
        name     : optional label
        corner_r : rounding radius applied at free corners

        Returns the PolygonPanel.  Use m.extend(panel, edge_idx, …) to grow
        child panels from any polygon edge.
        """
        if name is None:
            name = f'P{len(self.panels)}'
        p = PolygonPanel(origin, T, W, vertices, self.t, name, corner_r=corner_r)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def add_loft_panel(self, profiles, ruled=True, name=None):
        """Add a standalone LoftPanel (not connected to an existing panel).

        profiles : list of Curve3D or [(x,y,z),(x,y,z)] pairs
        ruled    : True → ruled surface; False → smooth loft
        name     : optional label

        Returns the LoftPanel.
        """
        if name is None:
            name = f'P{len(self.panels)}'
        p = LoftPanel(profiles=profiles, ruled=ruled, name=name, t=self.t)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def extend(self, parent, edge_idx, angle_deg, child_length,
               bend_dir=+1, start_frac=0.0, end_frac=1.0,
               blend_r=None, taper='arc'):
        child, arc = extend_panel(
            parent, edge_idx, angle_deg, child_length,
            inner_r=self.inner_r, bend_dir=bend_dir,
            start_frac=start_frac, end_frac=end_frac,
            blend_r=blend_r, taper=taper,
        )
        parent._register_child(edge_idx, start_frac, end_frac)
        parent_node = self._node_map[id(parent)]
        child_node  = PanelNode(child, parent=parent_node, edge_idx=edge_idx,
                                arc=arc, start_frac=start_frac, end_frac=end_frac)
        parent_node.children.append(child_node)
        self._node_map[id(child)] = child_node
        self.panels.append(child)
        self.arcs.append(arc)
        return child, arc

    def extend_loft(self, parent, edge_idx, angle_deg, free_curve,
                    bend_dir=+1, start_frac=0.0, end_frac=1.0,
                    intermediate_curves=None, ruled=True):
        """Extend a LoftPanel child from parent.

        Like extend() but creates a LoftPanel whose free edge is `free_curve`
        instead of a flat FlatPanel.

        Parameters
        ----------
        parent             : any panel type
        edge_idx           : edge on parent
        angle_deg          : bend angle in degrees
        free_curve         : Curve3D or [(x,y,z),(x,y,z)] for the far edge
        bend_dir           : +1 / -1
        start_frac, end_frac: span along parent edge
        intermediate_curves: optional list of Curve3D between root and free
        ruled              : True (default) or False for smooth loft

        Returns (LoftPanel child, arc).
        """
        child, arc = extend_loft(
            parent, edge_idx, angle_deg, free_curve,
            inner_r=self.inner_r, bend_dir=bend_dir,
            start_frac=start_frac, end_frac=end_frac,
            intermediate_curves=intermediate_curves, ruled=ruled,
        )
        parent._register_child(edge_idx, start_frac, end_frac)
        parent_node = self._node_map[id(parent)]
        child_node  = PanelNode(child, parent=parent_node, edge_idx=edge_idx,
                                arc=arc, start_frac=start_frac, end_frac=end_frac)
        parent_node.children.append(child_node)
        self._node_map[id(child)] = child_node
        self.panels.append(child)
        self.arcs.append(arc)
        return child, arc

    def add_recess(self, panel, cx, cy, width, height, corner_r,
                   r_top, r_bot, depth=None, wall_h=None,
                   draft_angle=0.0):
        """Stamp a drawn recess onto an existing panel.

        cx, cy      : centre of recess in panel local coords (from panel.origin,
                      along panel.T and panel.W)
        width       : outer footprint width  (along panel.T)
        height      : outer footprint height (along panel.W)
        corner_r    : rounded-rectangle corner radius (clamped to min(W,H)/2)
        r_top       : top bend radius
        r_bot       : bottom bend radius
        depth       : total recess depth (or specify wall_h)
        wall_h      : straight wall vertical height (derived from depth if omitted)
        draft_angle : wall slope angle from vertical in degrees (default 0)

        Returns the StampedRecess object.
        """
        rec = StampedRecess(panel, cx, cy, width, height, corner_r,
                            r_top, r_bot, depth=depth, wall_h=wall_h,
                            draft_angle=draft_angle)
        self.recesses.append(rec)
        return rec

    def add_polygon_recess(self, panel, polygon, corner_r=0.0,
                           r_top=3.0, r_bot=3.0, depth=None, wall_h=None,
                           draft_angle=0.0):
        """Stamp a drawn recess with a polygon footprint onto a panel.

        panel       : the host FlatPanel or PolygonPanel
        polygon     : (N, 2) list of (u, v) vertices in panel T-W local frame
        corner_r    : rounding radius at convex polygon corners (default 0)
        r_top       : top bend radius
        r_bot       : bottom bend radius
        depth       : total recess depth  (or specify wall_h)
        wall_h      : straight wall height
        draft_angle : wall tilt from panel-normal, degrees (default 0)

        Returns the PolygonRecess object.
        """
        rec = PolygonRecess(panel, polygon, corner_r=corner_r,
                            r_top=r_top, r_bot=r_bot,
                            depth=depth, wall_h=wall_h,
                            draft_angle=draft_angle)
        self.recesses.append(rec)
        return rec

    def node_of(self, panel):
        """Return the PanelNode for a given FlatPanel."""
        return self._node_map[id(panel)]

    def print_tree(self):
        """Print the panel connectivity tree to stdout."""
        for root in self.roots:
            print(root.format_tree())

    def mark_hole(self, panel, u, v, label='F', r_vis=5.0):
        centre = panel.origin + u*panel.T + v*panel.W
        self._holes.append({'centre': centre, 'normal': panel.N,
                            'label': label, 'r': r_vis})

