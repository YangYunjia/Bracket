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

from ..basics.maths import n, signed_area2, pt_in_tri
from ..basics.panel import FlatPanel, PolygonPanel, LoftPanel
from ..basics.bending import BendArc, PartialBendArc
from ..basics.recess import StampedRecess, PolygonRecess


def _ear_clip(verts):
    """Ear-clipping triangulation of a simple polygon.

    Returns list of (i, j, k) index triples into the original vertex array.
    Works for any simple polygon (convex or concave).
    """
    pts = np.asarray(verts, float)
    n   = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    flip = signed_area2(pts) < 0
    if flip:
        pts = pts[::-1]

    idx   = list(range(n))
    tris  = []
    guard = n * n

    while len(idx) > 3 and guard > 0:
        guard -= 1
        m       = len(idx)
        clipped = False
        for k in range(m):
            ia, ib, ic = (k-1) % m, k, (k+1) % m
            a, b, c = pts[idx[ia]], pts[idx[ib]], pts[idx[ic]]
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if cross <= 1e-10:
                continue
            if all(not pt_in_tri(pts[idx[j]], a, b, c)
                   for j in range(m) if j not in (ia, ib, ic)):
                tris.append((idx[ia], idx[ib], idx[ic]))
                idx.pop(ib)
                clipped = True
                break
        if not clipped:
            break

    if len(idx) == 3:
        tris.append(tuple(idx))

    if flip:
        tris = [(n-1-i, n-1-j, n-1-k) for i, j, k in tris]

    return tris


def _rounded_polygon_pts(vertices, radii, n_arc=12):
    """Discretise a CCW polygon with per-vertex corner rounding.

    Only convex vertices are rounded (concave vertices are always sharp).

    Parameters
    ----------
    vertices : (N, 2) CCW polygon vertices
    radii    : (N,) per-vertex rounding radii (0 = sharp)
    n_arc    : arc subdivisions per corner (higher = smoother)

    Returns
    -------
    list of 2-D np.array points (open ring: first ≠ last).
    """
    pts = np.asarray(vertices, float)
    n   = len(pts)
    R   = np.asarray(radii, float)

    # Per-vertex: trim length + arc geometry
    vtrim  = np.zeros(n)
    varc_p = [None] * n    # (center_2d, a_start, a_end, radius)

    for i in range(n):
        r = float(R[i])
        if r < 1e-12:
            continue
        v     = pts[i]
        e_in  = v - pts[(i-1) % n]
        e_out = pts[(i+1) % n] - v
        l_in  = np.linalg.norm(e_in)
        l_out = np.linalg.norm(e_out)
        if l_in < 1e-12 or l_out < 1e-12:
            continue
        e_in  /= l_in
        e_out /= l_out
        cross = e_in[0]*e_out[1] - e_in[1]*e_out[0]
        dot   = float(np.dot(e_in, e_out))
        if cross <= 1e-6:
            continue           # concave or collinear — skip
        sin_h = np.sqrt(max(0.0, (1.0 - dot) / 2.0))
        cos_h = np.sqrt(max(0.0, (1.0 + dot) / 2.0))
        if cos_h < 1e-9:
            continue
        trim  = r * sin_h / cos_h
        arc_s = v - trim * e_in
        arc_e = v + trim * e_out
        cen   = arc_s + r * np.array([-e_in[1], e_in[0]])
        a0    = np.arctan2(arc_s[1] - cen[1], arc_s[0] - cen[0])
        a1    = np.arctan2(arc_e[1] - cen[1], arc_e[0] - cen[0])
        if a1 <= a0:
            a1 += 2.0 * np.pi
        vtrim[i]  = trim
        varc_p[i] = (cen, a0, a1, r)

    # Build boundary point list
    result = []
    for i in range(n):
        j     = (i + 1) % n
        e_dir = pts[j] - pts[i]
        elen  = np.linalg.norm(e_dir)
        if elen < 1e-12:
            continue
        e_dir /= elen

        if vtrim[i] + vtrim[j] > elen + 1e-6:
            raise ValueError(
                f'Corner radii at vertices {i} and {j} are too large '
                f'(edge length {elen:.4g}, combined trim {vtrim[i]+vtrim[j]:.4g}).')

        p_s = pts[i] + vtrim[i] * e_dir   # line-segment start (arc_end of vertex i)
        p_e = pts[j] - vtrim[j] * e_dir   # line-segment end   (arc_start of vertex j)

        result.append(p_s)

        if not np.allclose(p_s, p_e, atol=1e-9):
            if varc_p[j] is not None:
                result.append(p_e)
            # else: p_e == vertices[j], added as p_s in the next iteration

        if varc_p[j] is not None:
            cen, a0, a1, r = varc_p[j]
            # k=0 is arc_start (= p_e, already added); k=n_arc is arc_end (= next p_s)
            for k in range(1, n_arc):
                a = a0 + k / n_arc * (a1 - a0)
                result.append(cen + r * np.array([np.cos(a), np.sin(a)]))

    return result

class NumpyBracketModel(SheetMetalModel):
    """SheetMetalModel + numpy/matplotlib rendering.

    Identical builder API to SheetMetalModel.  render() accepts optional
    quality keyword arguments to control tessellation resolution.
    """

    def __init__(self, t=2.0, inner_r=5.0,
               n_arc_bend=28, n_arc_partial=32, n_blend=16,
               n_sample=32, n_curve=12, n_boundary=80, n_profile=20):

        super().__init__(t=t, inner_r=inner_r)
        self.n_arc_bend = n_arc_bend
        self.n_arc_partial = n_arc_partial
        self.n_blend = n_blend
        self.n_sample = n_sample
        self.n_curve = n_curve
        self.n_boundary = n_boundary
        self.n_profile = n_profile
    
    # ---------------------------------------
    # FlatPanel rendering
    # ---------------------------------------

    def render_panel(self, obj: FlatPanel | PolygonPanel | LoftPanel):
        if isinstance(obj, FlatPanel):
            return self.render_flat_panel(obj)
        if isinstance(obj, PolygonPanel):
            return self.render_polygon_panel(obj)
        if isinstance(obj, LoftPanel):
            return self.render_loft_panel(obj)
        raise ValueError(f'Unknown panel type: {type(obj)}')

    def render_flat_panel(self, obj: FlatPanel):
        L, Wd = obj.length, obj.width
        if obj.corner_r > 0.0:
            verts_2d = np.array([[0, 0], [L, 0], [L, Wd], [0, Wd]], float)
            radii    = np.array([
                0.0 if obj._corner_blocked(i) else obj.corner_r
                for i in range(4)
            ])
            pts_2d = _rounded_polygon_pts(verts_2d, radii)
            pts_3d = [obj.origin + u*obj.T + v*obj.W for u, v in pts_2d]
            tris   = _ear_clip(np.array(pts_2d))
            return [[pts_3d[i], pts_3d[j], pts_3d[k]] for i, j, k in tris]
        C = obj.corners()
        return [[C[0], C[1], C[2], C[3]]]


    def render_polygon_panel(self, obj: PolygonPanel):
        """Ear-clipping triangulation → list of triangle vertex triples.

        Free corners (both adjacent edges unused) are rounded by corner_r.
        """
        if obj.corner_r > 0.0:
            n      = len(obj.vertices)
            radii  = np.array([
                0.0 if obj._corner_blocked(i) else obj.corner_r
                for i in range(n)
            ])
            pts_2d = _rounded_polygon_pts(obj.vertices, radii)
            pts_3d = [obj._pt3(u, v) for u, v in pts_2d]
            tris   = _ear_clip(np.array(pts_2d))
            return [[pts_3d[i], pts_3d[j], pts_3d[k]] for i, j, k in tris]
        tris = _ear_clip(obj.vertices)
        return [[obj._pt3(*obj.vertices[i]),
                 obj._pt3(*obj.vertices[j]),
                 obj._pt3(*obj.vertices[k])]
                for i, j, k in tris]
                

    def render_loft_panel(self, obj: LoftPanel):
        """Quad tessellation of the loft surface.

        For 2 profiles: directly interpolates between them.
        For N>2 profiles: interpolates between consecutive profile pairs.
        """
        sampled = [p.sample(self.n_sample) if p._kind != 'polyline'
                else p._pts
                for p in obj.profiles]
        quads = []

        for k in range(len(sampled) - 1):
            A = sampled[k]      # (n, 3) current profile
            B = sampled[k + 1]  # (n, 3) next profile
            n = self.n_sample
            # Ensure same count by resampling if needed (polyline case)
            if len(A) != n:
                ts = np.linspace(0.0, 1.0, n)
                idx_a = np.round(ts * (len(A) - 1)).astype(int)
                A = A[idx_a]
            if len(B) != n:
                ts = np.linspace(0.0, 1.0, n)
                idx_b = np.round(ts * (len(B) - 1)).astype(int)
                B = B[idx_b]
            for j in range(n - 1):
                quads.append([A[j], A[j+1], B[j+1], B[j]])
        return quads


    # ---------------------------------------
    # BendArc rendering
    # ---------------------------------------
    def render_arc(self, obj: BendArc | PartialBendArc):
        if isinstance(obj, BendArc):
            return self.render_bend_arc(obj)
        if isinstance(obj, PartialBendArc):
            return self.render_partial_bend_arc(obj)
        raise ValueError(f'Unknown arc type: {type(obj)}')

    @staticmethod
    def _pt(obj: BendArc | PartialBendArc, s, a):
        p_edge = (1-s)*obj.p0 + s*obj.p1
        C      = p_edge + obj.r_n * obj.N_bend
        return C + obj.r_n * (-np.cos(a)*obj.N_bend + np.sin(a)*obj.T_out)

    def render_bend_arc(self, obj: BendArc):
        alphas = np.linspace(0, obj.alpha, self.n_arc+1)
        return [[self._pt(obj, 0, alphas[i]),  self._pt(obj, 1, alphas[i]),
                 self._pt(obj, 1, alphas[i+1]), self._pt(obj, 0, alphas[i+1])]
                for i in range(self.n_arc)]
    
    @staticmethod
    def _ext(obj: PartialBendArc, a):
        t = a / obj.alpha
        if obj.taper == 'linear':
            scale = 1.0 - t
        else:
            scale = 1.0 - np.sqrt(max(0.0, t * (2 - t)))
        return obj.blend * scale

    def render_partial_bend_arc(self, obj: PartialBendArc):
        a_vals = np.linspace(0.0, obj.alpha, self.n_arc + 1)
        grid = []
        for a in a_vals:
            ext  = self._ext(obj, a)
            s_lo = max(0.0,   obj.s0 - ext)
            s_hi = min(obj.L, obj.s1 + ext)
            grid.append([self._pt(obj, np.clip(s / obj.L, 0.0, 1.0), a)
                         for s in np.linspace(s_lo, s_hi, self.n_blend + 1)])
        return [[grid[i][j], grid[i][j+1], grid[i+1][j+1], grid[i+1][j]]
                for i in range(self.n_arc) for j in range(self.n_blend)]

    # ---------------------------------------
    # Recess rendering
    # ---------------------------------------
    
    def render_recess(self, obj: StampedRecess | PolygonRecess):
        if isinstance(obj, StampedRecess):
            return obj.faces(n_boundary=self.n_boundary, n_profile=self.n_profile)
        if isinstance(obj, PolygonRecess):
            return obj.faces(n_boundary=self.n_boundary, n_profile=self.n_profile)
        raise ValueError(f'Unknown recess type: {type(obj)}')


    def render(self, ax=None,
               color_panel='#4a90c4', color_bend='#2e6ea6',
               color_recess='#3a7abf',
               alpha_surf=0.85, ec='#1a3a52', lw=0.4,
               elev=28., azim=-50.):
        """Render the model with matplotlib.

        Quality parameters (all optional):
          n_arc_bend, n_arc_partial, n_blend, n_sample,
          n_curve, n_boundary, n_profile — see module docstring.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        for p in self.panels:
            ax.add_collection3d(Poly3DCollection(
                self.render_panel(p), alpha=alpha_surf,
                facecolor=color_panel, edgecolor=ec, linewidth=lw))

        for arc in self.arcs:
            ax.add_collection3d(Poly3DCollection(
                self.render_arc(arc), alpha=alpha_surf,
                facecolor=color_bend, edgecolor=ec, linewidth=lw))

        for rec in self.recesses:
            ax.add_collection3d(Poly3DCollection(
                self.render_recess(rec), alpha=alpha_surf,
                facecolor=color_recess, edgecolor=ec, linewidth=lw))

        hole_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for i, h in enumerate(self._holes):
            c, ctr, n, r = hole_colors[i % 4], h['centre'], h['normal'], h['r']
            ref = np.array([0, 1, 0]) if abs(n[1]) < 0.85 else np.array([1, 0, 0])
            u_ = n(np.cross(n, ref))
            v_ = n(np.cross(n, u_))
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
