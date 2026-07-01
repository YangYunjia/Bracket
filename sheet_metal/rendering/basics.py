"""

Base class for rendering classes
"""

from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..basics.panel import FlatPanel, PolygonPanel, LoftPanel
from ..basics.bending import PartialBendArc
from ..basics.recess import StampedRecess, PolygonRecess
from ..basics.model import SheetMetalModel
from ..basics.maths import n

DEFAULT_GROUP_COLORS = {
    'panels': '#4a90c4',
    'arcs': '#2e6ea6',
    'recesses': '#9467bd',
    'holes': '#ff7f0e',
}

DEFAULT_MARKER_COLORS = {
    'holes': ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e'],
}

DEFAULT_EDGE_COLOR = '#1a3a52'


class BaseRenderer:

    def __init__(self, 
                 group_colors: dict=DEFAULT_GROUP_COLORS, 
                 marker_colors: dict=DEFAULT_MARKER_COLORS, 
                 edge_color: str=DEFAULT_EDGE_COLOR, 
                 alpha_surf: float=0.85,
                 lw: float=0.4
                 ) -> None:
        
        self.group_colors = group_colors
        self.marker_colors = marker_colors
        self.edge_color = edge_color
        self.alpha_surf = alpha_surf
        self.lw = lw

    @abstractmethod
    def render_panel(self, obj: FlatPanel | PolygonPanel | LoftPanel):
        pass

    @abstractmethod
    def render_arc(self, obj: PartialBendArc):
        pass

    @abstractmethod
    def render_recess(self, obj: StampedRecess | PolygonRecess):
        pass

    @abstractmethod
    def render(self, model: SheetMetalModel, **kwargs):
        groups = {
            'panels': {p.name: self.render_panel(p) for p in model.panels},
            'arcs': {a.name: self.render_arc(a) for a in model.arcs},
            'recesses': {r.name: self.render_recess(r) for r in model.recesses},
        }
        markers = {
            'holes': model._holes,
        }
        return groups, markers
        
    def export_plot(self, groups: dict, markers: dict, ax=None, elev=28., azim=-50.):
        """Render the model with matplotlib.

        Quality parameters (all optional):
          n_arc_bend, n_arc_partial, n_blend, n_sample,
          n_curve, n_boundary, n_profile — see module docstring.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        all_pts = []
        for group, objects in groups.items():
            for name, obj in objects.items():
                ax.add_collection3d(Poly3DCollection(
                    obj, alpha=self.alpha_surf,
                    facecolor=self.group_colors[group], edgecolor=self.edge_color, linewidth=self.lw))
                all_pts.extend(pt for tri in obj for pt in tri)

        for i, h in enumerate(markers['holes']):
            c, ctr, _n, r = self.marker_colors['holes'][i % 4], h['centre'], h['normal'], h['r']
            ref = np.array([0, 1, 0]) if abs(_n[1]) < 0.85 else np.array([1, 0, 0])
            u_ = n(np.cross(_n, ref))
            v_ = n(np.cross(_n, u_))
            ang  = np.linspace(0, 2 * np.pi, 48)
            ring = ctr + r * (np.outer(np.cos(ang), u_) + np.outer(np.sin(ang), v_))
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=c, lw=2.2, zorder=10)
            ax.quiver(*ctr, *(_n * r * 1.5), color=c, lw=1.8,
                      arrow_length_ratio=0.35, zorder=10)
            ax.text(*(ctr + _n * r * 2.5), h['label'],
                    fontsize=10, color=c, fontweight='bold')

        # Auto-scale
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

    @abstractmethod
    def export_step(self, groups: dict, filepath: str) -> None:
        pass