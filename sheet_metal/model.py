"""
sheet_metal.model — SheetMetalModel builder and renderer.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .primitives import _n, FlatPanel, BendArc, PartialBendArc, PolygonPanel
from .tree import PanelNode
from .recess import StampedRecess, PolygonRecess


def extend_panel(parent, edge_idx, angle_deg, child_length,
                 inner_r=5.0, bend_dir=+1,
                 start_frac=0.0, end_frac=1.0,
                 blend_r=None, taper='arc'):
    """Extend a child panel from one edge of a parent panel.

    Parameters
    ----------
    parent      : FlatPanel
    edge_idx    : 0–3  (see FlatPanel edge numbering)
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
    (child : FlatPanel, arc : BendArc | PartialBendArc)
    """
    p0_full, p1_full, edge_dir, T_out = parent.edge(edge_idx)
    L_edge = float(np.linalg.norm(p1_full - p0_full))

    p0 = p0_full + start_frac * (p1_full - p0_full)
    p1 = p0_full + end_frac   * (p1_full - p0_full)

    t     = parent.t
    alpha = np.radians(abs(angle_deg))
    bdir  = float(bend_dir) * parent.N
    r_n   = inner_r + t / 2

    T_child     = _n(np.sin(alpha)*bdir + np.cos(alpha)*T_out)
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

    def add_recess(self, panel, cx, cy, width, height, corner_r,
                   r_top, r_bot, depth=None, wall_h=None,
                   draft_angle=0.0,
                   n_boundary=80, n_profile=20):
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
                            draft_angle=draft_angle,
                            n_boundary=n_boundary, n_profile=n_profile)
        self.recesses.append(rec)
        return rec

    def add_polygon_recess(self, panel, polygon, corner_r=0.0,
                           r_top=3.0, r_bot=3.0, depth=None, wall_h=None,
                           draft_angle=0.0,
                           n_boundary=80, n_profile=20):
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
                            draft_angle=draft_angle,
                            n_boundary=n_boundary, n_profile=n_profile)
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

    def render(self, ax=None, color_panel='#4a90c4', color_bend='#2e6ea6',
               color_recess='#3a7abf',
               alpha_surf=0.85, ec='#1a3a52', lw=0.4, elev=28., azim=-50.):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        for p in self.panels:
            ax.add_collection3d(Poly3DCollection(
                p.faces(), alpha=alpha_surf,
                facecolor=color_panel, edgecolor=ec, linewidth=lw))

        for arc in self.arcs:
            ax.add_collection3d(Poly3DCollection(
                arc.faces(), alpha=alpha_surf,
                facecolor=color_bend, edgecolor=ec, linewidth=lw))

        for rec in self.recesses:
            ax.add_collection3d(Poly3DCollection(
                rec.faces(), alpha=alpha_surf,
                facecolor=color_recess, edgecolor=ec, linewidth=lw))

        hole_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for i, h in enumerate(self._holes):
            c, ctr, n, r = hole_colors[i % 4], h['centre'], h['normal'], h['r']
            ref = np.array([0, 1, 0]) if abs(n[1]) < 0.85 else np.array([1, 0, 0])
            u_ = _n(np.cross(n, ref))
            v_ = _n(np.cross(n, u_))
            ang  = np.linspace(0, 2*np.pi, 48)
            ring = ctr + r*(np.outer(np.cos(ang), u_) + np.outer(np.sin(ang), v_))
            ax.plot(ring[:,0], ring[:,1], ring[:,2], color=c, lw=2.2, zorder=10)
            ax.quiver(*ctr, *(n*r*1.5), color=c, lw=1.8,
                      arrow_length_ratio=0.35, zorder=10)
            ax.text(*(ctr + n*r*2.5), h['label'],
                    fontsize=10, color=c, fontweight='bold')

        pts = ([pt for p   in self.panels   for f in p.faces()   for pt in f] +
               [pt for arc in self.arcs    for f in arc.faces() for pt in f] +
               [pt for rec in self.recesses for f in rec.faces() for pt in f])
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        mid    = (lo + hi) / 2
        span   = (hi - lo).max() * 0.65
        for fn, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
            fn(m - span, m + span)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        return ax
