"""
sheet_metal.rendering.numpy_model — numpy/matplotlib rendering backend.

NumpyBracketModel wraps SheetMetalModel and adds render() with per-type
quality control parameters.  All quality values have sensible defaults so
the class can be used as a drop-in replacement for SheetMetalModel.

Quality parameters accepted by render():
  n_arc_bend    (int, 28)  — arc subdivisions for BendArc
  n_arc_partial (int, 32)  — arc subdivisions for PartialBendArc
  n_blend       (int, 16)  — width subdivisions for PartialBendArc
  n_sample      (int, 32)  — profile sample count for LoftPanel
  n_curve       (int, 12)  — edge samples for SplinePanel curved edges
  n_boundary    (int, 80)  — boundary resolution for recesses
  n_profile     (int, 20)  — cross-section resolution for recesses
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..basics.model import SheetMetalModel
from ..basics.primitives import (BendArc, PartialBendArc, CurvedBendArc,
                                  LoftPanel, SplinePanel)
from ..basics.recess import StampedRecess, PolygonRecess


def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v


def _get_faces(obj, n_arc_bend=28, n_arc_partial=32, n_blend=16,
               n_sample=32, n_curve=12, n_boundary=80, n_profile=20):
    """Dispatch to the correct faces() call with quality params."""
    if isinstance(obj, PartialBendArc):
        return obj.faces(n_arc=n_arc_partial, n_blend=n_blend)
    if isinstance(obj, BendArc):
        return obj.faces(n_arc=n_arc_bend)
    if isinstance(obj, CurvedBendArc):
        return obj.faces()   # uses instance n_edge / n_arc (shared with OCC)
    if isinstance(obj, LoftPanel):
        return obj.faces(n_sample=n_sample)
    if isinstance(obj, SplinePanel):
        return obj.faces(n_curve=n_curve)
    if isinstance(obj, StampedRecess):
        return obj.faces(n_boundary=n_boundary, n_profile=n_profile)
    if isinstance(obj, PolygonRecess):
        return obj.faces(n_boundary=n_boundary, n_profile=n_profile)
    return obj.faces()


class NumpyBracketModel(SheetMetalModel):
    """SheetMetalModel + numpy/matplotlib rendering.

    Identical builder API to SheetMetalModel.  render() accepts optional
    quality keyword arguments to control tessellation resolution.
    """

    def render(self, ax=None,
               color_panel='#4a90c4', color_bend='#2e6ea6',
               color_recess='#3a7abf',
               alpha_surf=0.85, ec='#1a3a52', lw=0.4,
               elev=28., azim=-50.,
               n_arc_bend=28, n_arc_partial=32, n_blend=16,
               n_sample=32, n_curve=12, n_boundary=80, n_profile=20):
        """Render the model with matplotlib.

        Quality parameters (all optional):
          n_arc_bend, n_arc_partial, n_blend, n_sample,
          n_curve, n_boundary, n_profile — see module docstring.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        q = dict(n_arc_bend=n_arc_bend, n_arc_partial=n_arc_partial,
                 n_blend=n_blend, n_sample=n_sample, n_curve=n_curve,
                 n_boundary=n_boundary, n_profile=n_profile)

        for p in self.panels:
            ax.add_collection3d(Poly3DCollection(
                _get_faces(p, **q), alpha=alpha_surf,
                facecolor=color_panel, edgecolor=ec, linewidth=lw))

        for arc in self.arcs:
            ax.add_collection3d(Poly3DCollection(
                _get_faces(arc, **q), alpha=alpha_surf,
                facecolor=color_bend, edgecolor=ec, linewidth=lw))

        for rec in self.recesses:
            ax.add_collection3d(Poly3DCollection(
                _get_faces(rec, **q), alpha=alpha_surf,
                facecolor=color_recess, edgecolor=ec, linewidth=lw))

        hole_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for i, h in enumerate(self._holes):
            c, ctr, n, r = hole_colors[i % 4], h['centre'], h['normal'], h['r']
            ref = np.array([0, 1, 0]) if abs(n[1]) < 0.85 else np.array([1, 0, 0])
            u_ = _n(np.cross(n, ref))
            v_ = _n(np.cross(n, u_))
            ang  = np.linspace(0, 2 * np.pi, 48)
            ring = ctr + r * (np.outer(np.cos(ang), u_) + np.outer(np.sin(ang), v_))
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=c, lw=2.2, zorder=10)
            ax.quiver(*ctr, *(n * r * 1.5), color=c, lw=1.8,
                      arrow_length_ratio=0.35, zorder=10)
            ax.text(*(ctr + n * r * 2.5), h['label'],
                    fontsize=10, color=c, fontweight='bold')

        all_pts = ([pt for p   in self.panels   for f in _get_faces(p,   **q) for pt in f] +
                   [pt for arc in self.arcs     for f in _get_faces(arc, **q) for pt in f] +
                   [pt for rec in self.recesses for f in _get_faces(rec, **q) for pt in f])
        if all_pts:
            pts   = np.array(all_pts)
            lo, hi = pts.min(0), pts.max(0)
            mid   = (lo + hi) / 2
            span  = max((hi - lo).max() * 0.65, 1.0)
            for fn, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
                fn(m - span, m + span)

        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        return ax
