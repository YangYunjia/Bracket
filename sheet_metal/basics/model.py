"""
sheet_metal.model — SheetMetalModel builder and renderer.
"""

import numpy as np

from .maths import n
from .curve import Curve3D, BaseCurve3D
from .panel import Panel, FlatPanel, PolygonPanel, LoftPanel, PipePanel
from .bending import PartialBendArc
from .tree import PanelNode
from .recess import StampedRecess, PolygonRecess

from typing import List

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
        self.panels: List[Panel]    = []        # all FlatPanel objects in insertion order
        self.arcs: List[PartialBendArc]      = []        # all arc objects in insertion order
        self.recesses: List[StampedRecess | PolygonRecess]  = []        # all StampedRecess objects
        self._holes    = []
        self.roots     = []        # root PanelNodes (one per add_panel call)
        self._node_map = {}        # id(FlatPanel) → PanelNode

    def add_panel(self, name, origin, T, W, length, width, corner_r=0.0):
        p = FlatPanel(name, origin, T, W, length, width, corner_r=corner_r)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def add_polygon_panel(self, name, origin, T, W, vertices, corner_r=0.0):
        """Add a flat panel with an arbitrary polygon outline.

        origin   : 3-D position of the panel coordinate origin
        T, W     : two orthogonal 3-D vectors spanning the panel plane
        vertices : (N, 2) list/array of (u, v) pairs in the T-W local frame
        name     : optional label
        corner_r : rounding radius applied at free corners

        Returns the PolygonPanel.  Use m.extend(panel, edge_idx, …) to grow
        child panels from any polygon edge.
        """
        p = PolygonPanel(name, origin, T, W, vertices, corner_r=corner_r)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def add_loft_panel(self, name, profiles, ruled=True):
        """Add a standalone LoftPanel (not connected to an existing panel).

        profiles : list of Curve3D or [(x,y,z),(x,y,z)] pairs
        ruled    : True → ruled surface; False → smooth loft
        name     : optional label

        Returns the LoftPanel.
        """
        if name is None:
            name = f'P{len(self.panels)}'
        p = LoftPanel(name, profiles, ruled=ruled)
        node = PanelNode(p)
        self.panels.append(p)
        self.roots.append(node)
        self._node_map[id(p)] = node
        return p

    def extend_bend(self, parent: PolygonPanel | LoftPanel, edge_idx, angle_deg,
               bend_dir=+1, bend_r=5, start_frac=0.0, end_frac=1.0,
               blend_r=0.5, blend_mode='arc',
               child_name='0') -> PartialBendArc:
        """
        Extend a panel from an edge via a bend arc.

        Parameters
        ----------
        parent : PolygonPanel | LoftPanel
        edge_idx : int (for LoftPanel, only 0 and 2 are extendable)
        angle_deg : float (bend angle in degrees)
        bend_dir : int (1 or -1)
        bend_r : float (bend radius)
        start_frac : float (without the blend_r)
        end_frac : float (without the blend_r)
        blend_r : float (blend radius on each side of the edge)
        blend_mode : str (blend mode, 'arc' or 'linear')
        child_name : str (name of the child panel)
        Returns
        -------
        arc : PartialBendArc
        raises ValueError if the edge is occupied by another child panel
        """
        if isinstance(parent, LoftPanel) and edge_idx not in LoftPanel._EXTENDABLE_EDGES:
            raise ValueError(
                f'LoftPanel only supports extend on profile edges '
                f'{LoftPanel._EXTENDABLE_EDGES} (root=0, free=2), got {edge_idx}.')

        # transform start_frac and end_frac to length
        crv = parent.edge_curve(edge_idx)
        L = crv.length()
        edge_dir = crv.tangent_at_frac(0.5)
        N_src = (parent.N_at_edge(edge_idx)
                 if hasattr(parent, 'N_at_edge')
                 else parent.N)
        T_out = n(np.cross(crv.tangent_at_frac(0.0), N_src))
        start_frac_length = start_frac * L
        end_frac_length   = (1 - end_frac) * L
        _st = parent.start_idx(edge_idx)
        _en = parent.end_idx(edge_idx)

        # downgrade edge corner_trim and blend_r if conflicts with the new panel (plus blend_r)
        # this includes the case without blend_r and corner_trim
        if start_frac_length - blend_r < parent.corner_trim[_st]:
            _st_blend_r = 0.5 * start_frac_length
            parent.corner_trim[_st] = _st_blend_r
        else:
            _st_blend_r = blend_r
        
        if end_frac_length - blend_r < parent.corner_trim[_en]:
            _en_blend_r = 0.5 * end_frac_length
            parent.corner_trim[_en] = _en_blend_r
        else:
            _en_blend_r = blend_r

        # TODO: reduce blend_r if conflicts with the new panel

        # Occupied span includes blend taper on each side of [start_frac, end_frac]:
        #   [start_frac·L − blend_start,  end_frac·L + blend_end]  (clamped to edge)
        st_frac_blend = max(0.0, (start_frac_length - _st_blend_r) / L)
        en_frac_blend = min(1.0, 1.0 - (end_frac_length - _en_blend_r) / L)
        if parent.is_edge_occupied(edge_idx, st_frac_blend, en_frac_blend):
            raise ValueError(f"Edge {edge_idx} is occupied by another child panel")  

        parent._register_child(edge_idx, st_frac_blend, en_frac_blend)
        parent.split_edge_at_fracs(edge_idx, [st_frac_blend, en_frac_blend])
        
        alpha = np.radians(abs(angle_deg))
        bend_dir  = float(bend_dir) * N_src

        p0 = crv.interpolate(start_frac)
        p1 = crv.interpolate(end_frac)
        edge_dir = crv.tangent_at_frac((start_frac + end_frac) * 0.5)
        T_out = n(np.cross(crv.tangent_at_frac(start_frac), N_src))
        arc = PartialBendArc(
            name=f'{parent.name}-{child_name}-arc', p0=p0, p1=p1,
            T_in=T_out, W_in=edge_dir, N_bend=bend_dir,
            alpha=alpha, r_n=bend_r,
            blend_r=(_st_blend_r, _en_blend_r), blend_mode=blend_mode,
        )

        self.arcs.append(arc)
        return arc

    def register_in_tree(self, parent, edge_idx, child, arc, start_frac, end_frac):
        parent_node = self._node_map[id(parent)]
        child_node  = PanelNode(child, parent=parent_node, edge_idx=edge_idx,
                                arc=arc, start_frac=start_frac, end_frac=end_frac)
        parent_node.children.append(child_node)
        self._node_map[id(child)] = child_node
        return child, arc

    def extend(self, parent: PolygonPanel | LoftPanel, edge_idx, angle_deg, 
               panel_type='flat',
               child_length=10.0, child_corner_r=0.0,
               free_curve=None, intermediate_curves=None, t_end=None, ruled=False,
               bend_dir=+1, bend_r=5, start_frac=0.0, end_frac=1.0,
               blend_r=0.5, blend_mode='arc', 
               name=None):
            
        if name is None:
            name = f'{parent.name}.e{edge_idx}'

        arc = self.extend_bend(parent, edge_idx, angle_deg,
                              bend_dir=bend_dir, bend_r=bend_r, start_frac=start_frac, end_frac=end_frac,
                              blend_r=blend_r, blend_mode=blend_mode, child_name=name)
        if panel_type == 'flat':
            child = self.extend_flat_panel(name, arc, child_length, child_corner_r)
        elif panel_type in ['loft', 'pipe']:
            child = self.extend_loft_panel(name, arc, free_curve, intermediate_curves, t_end, ruled)
        else:
            raise ValueError(f"Invalid panel type: {panel_type}")

        self.register_in_tree(parent, edge_idx, child, arc, start_frac, end_frac)
        
        return child, arc

    def extend_flat_panel(self, name: str, arc: PartialBendArc, child_length, child_corner_r=0.0):

        # Child panel direction ──────
        child = FlatPanel(
            name=name,
            origin=arc.p0_out, T=arc.T_out, W=arc.W,
            length=child_length, width=arc.L,
            corner_r=child_corner_r,
        )
        child.corner_trim[0] = 0.0
        child.corner_trim[1] = 0.0
        child._register_child(0, 0.0, 1.0)
        # print("Child panel added at edge", edge_idx, "with origin:", child_origin)
        self.panels.append(child)

        return child


    def extend_loft_panel(self, name: str, arc: PartialBendArc, free_curve,
                   intermediate_curves=None, t_end=None, ruled=False, panel_type='loft'):
        """Extend a LoftPanel or PipePanel child from parent.

        Parameters
        ----------
        name               : name of the panel
        arc                : PartialBendArc
        free_curve         : Curve3D or [(x,y,z),(x,y,z)] for the far edge
        intermediate_curves: optional list of Curve3D between root and free
        t_end              : optional unit tangent override at the free edge
        ruled              : True → ruled; False → smooth (default)

        Returns the LoftPanel / PipePanel.
        """
        
        root_curve = Curve3D.line(arc.p0_out, arc.p1_out)

        if not isinstance(free_curve, BaseCurve3D):
            pts = np.asarray(free_curve, float)
            free_curve = (Curve3D.line(pts[0], pts[1])
                        if pts.ndim == 2 and len(pts) == 2
                        else Curve3D.polyline(pts))

        profiles = ([root_curve]
                    + (list(intermediate_curves) if intermediate_curves else [])
                    + [free_curve])

        if t_end is None:
            a_root, a_free = root_curve.endpoints()[0], free_curve.endpoints()[0]
            t_end_val = n(a_free - a_root)
        else:
            t_end_val = n(np.asarray(t_end, float))

        kwargs = dict(name=name, profiles=profiles, t_start=arc.T_out, t_end=t_end_val,
                    ruled=ruled)
        child = LoftPanel(**kwargs) if panel_type == 'loft' else PipePanel(**kwargs)
        
        child.corner_trim[0] = 0.0
        child.corner_trim[1] = 0.0
        child._register_child(0, 0.0, 1.0)
        self.panels.append(child)

        return child


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

