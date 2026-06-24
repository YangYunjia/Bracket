"""
sheet_metal.primitives — geometry primitives for mid-surface shell models.

FlatPanel      : flat rectangular panel
BendArc        : full-edge cylindrical arc strip
PartialBendArc : partial-edge arc with smooth width taper
PolygonPanel   : flat panel with arbitrary polygon outline
BSplineEdge    : cubic B-spline curve for one panel edge
SplinePanel    : flat panel with curved (B-spline) edge outlines
CurvedBendArc  : arc strip that sweeps along a curved (B-spline) edge
"""

import numpy as np


def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v


# ── polygon helpers ───────────────────────────────────────────────────────────

def _signed_area2(pts):
    """Twice the signed area (positive = CCW)."""
    n = len(pts)
    return float(sum(pts[i, 0]*pts[(i+1)%n, 1] - pts[(i+1)%n, 0]*pts[i, 1]
                     for i in range(n)))


def _pt_in_tri(p, a, b, c):
    """True if point p is strictly inside triangle (a, b, c)."""
    def _s(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = _s(p, a, b), _s(p, b, c), _s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


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

    flip = _signed_area2(pts) < 0
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
            if all(not _pt_in_tri(pts[idx[j]], a, b, c)
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


# ── FlatPanel ─────────────────────────────────────────────────────────────────

class FlatPanel:
    """Flat rectangular panel (mid-surface only).

    Panel edge numbering
    ────────────────────
        C3 ─── e3 (v=W_) ─── C2
        │  e0 (u=0)  e1 (u=L)  │
        C0 ─── e2 (v=0)  ─── C1

      e0  C0→C3   outward = -T
      e1  C1→C2   outward = +T
      e2  C0→C1   outward = -W
      e3  C3→C2   outward = +W

    corner_r : radius for free corners (corners whose two adjacent edges are
               not connected to child panels).  0 = sharp corners everywhere.
    """

    # Per corner: which end of each adjacent edge the corner lies at.
    # (edge_idx, at_start) where at_start=True → corner is at frac=0 of that edge.
    # C0=(0,0): start of e0 (e0:C0→C3), start of e2 (e2:C0→C1)
    # C1=(L,0): start of e1 (e1:C1→C2), end   of e2
    # C2=(L,W): end   of e1,             end   of e3 (e3:C3→C2)
    # C3=(0,W): end   of e0,             start of e3
    _CORNER_EDGE_ENDS = [
        ((0, True),  (2, True)),
        ((1, True),  (2, False)),
        ((1, False), (3, False)),
        ((0, False), (3, True)),
    ]

    # Which FlatPanel corner is at the start and end of each edge.
    # e0: C0→C3, e1: C1→C2, e2: C0→C1, e3: C3→C2
    _EDGE_CORNERS = [(0, 3), (1, 2), (0, 1), (3, 2)]

    def __init__(self, origin, T, W, length, width, t=2.0, name='P',
                 corner_r=0.0):
        self.origin      = np.asarray(origin, float)
        self.T           = _n(T)
        self.W           = _n(W)
        self.length      = float(length)
        self.width       = float(width)
        self.t           = float(t)
        self.name        = str(name)
        self.corner_r    = float(corner_r)
        self._edge_spans = {}   # edge_idx → [(start_frac, end_frac), ...]

    def _register_child(self, edge_idx, start_frac=0.0, end_frac=1.0):
        """Record a child panel spanning [start_frac, end_frac] on this edge.

        Raises ValueError if the new span overlaps any existing span.
        """
        sf, ef = float(start_frac), float(end_frac)
        spans  = self._edge_spans.setdefault(edge_idx, [])
        for s, e in spans:
            if sf < e - 1e-6 and ef > s + 1e-6:
                raise ValueError(
                    f'Child [{sf:.3g},{ef:.3g}] overlaps existing child '
                    f'[{s:.3g},{e:.3g}] on edge {edge_idx}.')
        spans.append((sf, ef))

    def _corner_blocked(self, i):
        """Return True if corner i must not be rounded (a child touches it)."""
        for edge_idx, at_start in self._CORNER_EDGE_ENDS[i]:
            for sf, ef in self._edge_spans.get(edge_idx, []):
                if at_start and sf < 1e-6:
                    return True
                if not at_start and ef > 1.0 - 1e-6:
                    return True
        return False

    def _edge_flat_trims(self, edge_idx):
        """Return (trim_start, trim_end) distances that the flat region of edge
        edge_idx is set back from each corner by corner rounding.

        A trim is non-zero only when corner_r > 0 and the corner at that end
        is free (not blocked by any registered child).  All FlatPanel corners
        are 90°, so the trim equals corner_r.
        """
        if self.corner_r == 0.0:
            return 0.0, 0.0
        c_start, c_end = self._EDGE_CORNERS[edge_idx]
        ts = 0.0 if self._corner_blocked(c_start) else self.corner_r
        te = 0.0 if self._corner_blocked(c_end)   else self.corner_r
        return ts, te

    @property
    def N(self):
        return _n(np.cross(self.T, self.W))

    def corners(self):
        o, T, W, L, Wd = self.origin, self.T, self.W, self.length, self.width
        return [o, o+L*T, o+L*T+Wd*W, o+Wd*W]   # C0,C1,C2,C3

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for edge i."""
        C = self.corners()
        T, W = self.T, self.W
        data = [
            (C[0], C[3],  W, -T),
            (C[1], C[2],  W,  T),
            (C[0], C[1],  T, -W),
            (C[3], C[2],  T,  W),
        ]
        p0, p1, ed, out = data[i]
        return p0.copy(), p1.copy(), _n(ed), _n(out)

    def faces(self):
        L, Wd = self.length, self.width
        if self.corner_r > 0.0:
            verts_2d = np.array([[0, 0], [L, 0], [L, Wd], [0, Wd]], float)
            radii    = np.array([
                0.0 if self._corner_blocked(i) else self.corner_r
                for i in range(4)
            ])
            pts_2d = _rounded_polygon_pts(verts_2d, radii)
            pts_3d = [self.origin + u*self.T + v*self.W for u, v in pts_2d]
            tris   = _ear_clip(np.array(pts_2d))
            return [[pts_3d[i], pts_3d[j], pts_3d[k]] for i, j, k in tris]
        C = self.corners()
        return [[C[0], C[1], C[2], C[3]]]


# ── BendArc ───────────────────────────────────────────────────────────────────

class BendArc:
    """Full-edge cylindrical arc (mid-surface strip)."""

    def __init__(self, p0, p1, T_out, N_bend, alpha, r_n, n_arc=28):
        self.p0     = np.asarray(p0, float)
        self.p1     = np.asarray(p1, float)
        self.T_out  = _n(T_out)
        self.N_bend = _n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        self.n_arc  = int(n_arc)

    def _pt(self, s, a):
        p_edge = (1-s)*self.p0 + s*self.p1
        C      = p_edge + self.r_n * self.N_bend
        return C + self.r_n * (-np.cos(a)*self.N_bend + np.sin(a)*self.T_out)

    def faces(self):
        alphas = np.linspace(0, self.alpha, self.n_arc+1)
        return [[self._pt(0, alphas[i]),  self._pt(1, alphas[i]),
                 self._pt(1, alphas[i+1]), self._pt(0, alphas[i+1])]
                for i in range(self.n_arc)]


# ── PartialBendArc ────────────────────────────────────────────────────────────

class PartialBendArc:
    """Partial-edge cylindrical arc with a smooth width taper.

    One continuous bending surface — no separate pieces.

    The child panel connects along [s_start, s_end].  The arc widens at the
    base (a=0) by blend_r on each side, giving an effective connection length
    of child_width + 2·blend_r at the parent edge.  The width tapers back to
    child_width at a=alpha via ext(a) = blend_r · f(a/alpha), where f depends
    on the taper mode:

        'arc'    (default) — normalized quarter-circle:  f(t) = 1 − √(t·(2−t))
        'linear'           — straight ramp:               f(t) = 1 − t

    Both guarantee ext(0) = blend_r and ext(alpha) = 0.

    The s-range at arc angle a is [s_start − ext(a), s_end + ext(a)],
    clamped to the parent edge.  The surface is tessellated as an
    (n_arc × n_blend) quad grid.
    """

    def __init__(self, p0_full, p1_full,
                 s_start, s_end,
                 T_out, N_bend,
                 alpha, r_n,
                 blend_r=None, taper='arc', n_arc=32, n_blend=16):
        self.p0     = np.asarray(p0_full, float)
        self.p1     = np.asarray(p1_full, float)
        self.L      = float(np.linalg.norm(self.p1 - self.p0))
        self.s0     = float(s_start)
        self.s1     = float(s_end)
        self.T_out  = _n(T_out)
        self.N_bend = _n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        self.blend  = float(blend_r) if blend_r is not None else float(r_n)
        self.taper  = taper
        self.n_arc  = int(n_arc)
        self.n_bld  = int(n_blend)

    def _edge_pt(self, s):
        frac = np.clip(s / self.L, 0.0, 1.0)
        return (1 - frac)*self.p0 + frac*self.p1

    def _pt(self, s, a):
        C = self._edge_pt(s) + self.r_n * self.N_bend
        return C + self.r_n * (-np.cos(a)*self.N_bend + np.sin(a)*self.T_out)

    def _ext(self, a):
        t = a / self.alpha
        if self.taper == 'linear':
            scale = 1.0 - t
        else:
            scale = 1.0 - np.sqrt(max(0.0, t * (2 - t)))
        return self.blend * scale

    def faces(self):
        na = self.n_arc
        ns = self.n_bld
        a_vals = np.linspace(0.0, self.alpha, na + 1)
        grid = []
        for a in a_vals:
            ext  = self._ext(a)
            s_lo = max(0.0,    self.s0 - ext)
            s_hi = min(self.L, self.s1 + ext)
            grid.append([self._pt(s, a)
                         for s in np.linspace(s_lo, s_hi, ns + 1)])
        return [[grid[i][j], grid[i][j+1], grid[i+1][j+1], grid[i+1][j]]
                for i in range(na) for j in range(ns)]


# ── PolygonPanel ──────────────────────────────────────────────────────────────

class PolygonPanel:
    """Flat panel with an arbitrary polygon outline.

    The panel lives in the plane spanned by T and W from origin.
    Vertices are (u, v) pairs in that local frame and are normalised
    internally to CCW winding.

    Edge numbering: edge i goes from vertex[i] to vertex[(i+1) % n].
    The outward normal of each edge is the right-hand perpendicular of the
    edge direction (correct for CCW polygons).

    corner_r : radius for free corners (corners whose two adjacent edges are
               not connected to child panels).  0 = sharp everywhere.
    """

    def __init__(self, origin, T, W, vertices, t=2.0, name='P',
                 corner_r=0.0):
        """
        Parameters
        ----------
        origin   : 3-D anchor point
        T, W     : two orthogonal 3-D vectors spanning the panel plane
        vertices : (N, 2) array-like of (u, v) pairs, N >= 3
        t        : sheet thickness
        name     : label string
        corner_r : rounding radius for free corners
        """
        self.origin      = np.asarray(origin, float)
        self.T           = _n(T)
        self.W           = _n(W)
        self.t           = float(t)
        self.name        = str(name)
        self.corner_r    = float(corner_r)
        self._edge_spans = {}

        pts = np.asarray(vertices, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError('vertices must be (N>=3, 2) array of (u, v) pairs.')
        if _signed_area2(pts) < 0:
            pts = pts[::-1]        # normalise to CCW
        self.vertices = pts        # (N, 2)

    @property
    def N(self):
        return _n(np.cross(self.T, self.W))

    def _pt3(self, u, v):
        return self.origin + float(u)*self.T + float(v)*self.W

    @property
    def num_edges(self):
        return len(self.vertices)

    def _register_child(self, edge_idx, start_frac=0.0, end_frac=1.0):
        """Record a child panel spanning [start_frac, end_frac] on this edge.

        Raises ValueError if the new span overlaps any existing span.
        """
        sf, ef = float(start_frac), float(end_frac)
        spans  = self._edge_spans.setdefault(edge_idx, [])
        for s, e in spans:
            if sf < e - 1e-6 and ef > s + 1e-6:
                raise ValueError(
                    f'Child [{sf:.3g},{ef:.3g}] overlaps existing child '
                    f'[{s:.3g},{e:.3g}] on edge {edge_idx}.')
        spans.append((sf, ef))

    def _corner_blocked(self, i):
        """Return True if vertex i must not be rounded (a child touches it).

        Vertex i is at the end of edge (i-1)%n and the start of edge i.
        """
        n = len(self.vertices)
        for sf, ef in self._edge_spans.get((i - 1) % n, []):
            if ef > 1.0 - 1e-6:
                return True
        for sf, ef in self._edge_spans.get(i, []):
            if sf < 1e-6:
                return True
        return False

    def _corner_trim(self, i):
        """Compute the corner-rounding trim length at vertex i.

        Returns 0.0 if corner_r == 0, the vertex is concave, or the corner is
        blocked by a registered child.  Otherwise uses the half-angle formula:
            trim = corner_r * tan(φ/2)
        where φ is the exterior deflection angle at vertex i.
        """
        if self.corner_r == 0.0 or self._corner_blocked(i):
            return 0.0
        n = len(self.vertices)
        v     = self.vertices[i]
        e_in  = v - self.vertices[(i - 1) % n]
        e_out = self.vertices[(i + 1) % n] - v
        l_in, l_out = np.linalg.norm(e_in), np.linalg.norm(e_out)
        if l_in < 1e-12 or l_out < 1e-12:
            return 0.0
        e_in  /= l_in;  e_out /= l_out
        cross = e_in[0]*e_out[1] - e_in[1]*e_out[0]
        if cross <= 1e-6:
            return 0.0
        dot   = float(np.dot(e_in, e_out))
        sin_h = np.sqrt(max(0.0, (1.0 - dot) / 2.0))
        cos_h = np.sqrt(max(0.0, (1.0 + dot) / 2.0))
        if cos_h < 1e-9:
            return 0.0
        return self.corner_r * sin_h / cos_h

    def _edge_flat_trims(self, edge_idx):
        """Return (trim_start, trim_end) for edge edge_idx.

        edge_idx goes from vertex edge_idx to vertex (edge_idx+1)%n.
        """
        if self.corner_r == 0.0:
            return 0.0, 0.0
        n = len(self.vertices)
        return self._corner_trim(edge_idx), self._corner_trim((edge_idx + 1) % n)

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for polygon edge i.

        Compatible with extend_panel / SheetMetalModel.extend().
        """
        verts = self.vertices
        n     = len(verts)
        u0, v0 = verts[i % n]
        u1, v1 = verts[(i + 1) % n]
        p0 = self._pt3(u0, v0)
        p1 = self._pt3(u1, v1)
        du, dv = u1 - u0, v1 - v0
        L = np.hypot(du, dv)
        if L < 1e-12:
            raise ValueError(f'Polygon edge {i} has zero length.')
        # Edge direction in 3-D
        ed  = (du/L)*self.T + (dv/L)*self.W
        # Outward normal: right-hand perp of CCW edge = rotate −90°
        out = (dv/L)*self.T + (-du/L)*self.W
        return p0.copy(), p1.copy(), _n(ed), _n(out)

    def faces(self):
        """Ear-clipping triangulation → list of triangle vertex triples.

        Free corners (both adjacent edges unused) are rounded by corner_r.
        """
        if self.corner_r > 0.0:
            n      = len(self.vertices)
            radii  = np.array([
                0.0 if self._corner_blocked(i) else self.corner_r
                for i in range(n)
            ])
            pts_2d = _rounded_polygon_pts(self.vertices, radii)
            pts_3d = [self._pt3(u, v) for u, v in pts_2d]
            tris   = _ear_clip(np.array(pts_2d))
            return [[pts_3d[i], pts_3d[j], pts_3d[k]] for i, j, k in tris]
        tris = _ear_clip(self.vertices)
        return [[self._pt3(*self.vertices[i]),
                 self._pt3(*self.vertices[j]),
                 self._pt3(*self.vertices[k])]
                for i, j, k in tris]


# ── B-spline edge ─────────────────────────────────────────────────────────────

class BSplineEdge:
    """Cubic B-spline curve for one panel edge, in panel-local (u, v) coordinates.

    Control points must start at the edge's first vertex and end at the second:
    eval(0) == control_pts[0], eval(1) == control_pts[-1].

    A clamped uniform knot vector is generated automatically.  For 4 control
    points this is equivalent to a cubic Bézier curve.
    """

    def __init__(self, control_pts, degree=3):
        from scipy.interpolate import BSpline as _BSpline
        cp = np.asarray(control_pts, float)
        if cp.ndim != 2 or cp.shape[1] != 2:
            raise ValueError('control_pts must be (M, 2).')
        m = len(cp)
        if m < degree + 1:
            raise ValueError(
                f'Need at least {degree + 1} control points for degree {degree}.')
        self._cp     = cp
        self._degree = degree
        # clamped uniform knot vector
        n_internal = m - degree - 1
        interior   = (np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
                      if n_internal > 0 else np.empty(0))
        knots = np.concatenate([
            np.zeros(degree + 1),
            interior,
            np.ones(degree + 1),
        ])
        self._spu  = _BSpline(knots, cp[:, 0], degree)
        self._spv  = _BSpline(knots, cp[:, 1], degree)
        self._dspu = self._spu.derivative()
        self._dspv = self._spv.derivative()

    def eval(self, t):
        """Evaluate the curve at parameter t ∈ [0, 1] (scalar or array)."""
        t = np.asarray(t, float)
        return np.stack([self._spu(t), self._spv(t)], axis=-1)

    def tangent(self, t):
        """Un-normalised tangent (derivative) at parameter t."""
        t = np.asarray(t, float)
        return np.stack([self._dspu(t), self._dspv(t)], axis=-1)

    def arc_length(self, n=256):
        """Approximate arc length via dense sample sum."""
        pts = self.eval(np.linspace(0.0, 1.0, n + 1))
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def sample(self, n):
        """Return n uniformly-spaced-in-t samples including endpoints."""
        return self.eval(np.linspace(0.0, 1.0, n))

    def sample_arclength(self, n):
        """Return n samples evenly spaced in arc length."""
        n_dense  = max(512, n * 16)
        ts_dense = np.linspace(0.0, 1.0, n_dense)
        pts      = self.eval(ts_dense)
        cum      = np.concatenate([[0.0],
                                   np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        total    = cum[-1]
        if total < 1e-12:
            idx = np.round(np.linspace(0, n_dense - 1, n)).astype(int)
            return pts[idx]
        ts_out = np.interp(np.linspace(0.0, total, n), cum, ts_dense)
        return self.eval(ts_out)

    def reverse(self):
        """Return a new BSplineEdge traversing from eval(1) to eval(0)."""
        return BSplineEdge(self._cp[::-1], self._degree)

    @property
    def control_pts(self):
        return self._cp.copy()

    @property
    def degree(self):
        return self._degree


# ── SplinePanel ───────────────────────────────────────────────────────────────

class SplinePanel:
    """Flat panel with an arbitrary polygon outline where edges can be B-spline curves.

    Interface is identical to PolygonPanel so SplinePanel works transparently
    with extend_panel() and SheetMetalModel.extend().

    Parameters
    ----------
    origin      : 3-D anchor point
    T, W        : orthonormal basis vectors spanning the panel plane
    vertices    : (N, 2) CCW polygon corners in the T-W local frame
    edge_curves : list of N items (BSplineEdge | None).  None → straight edge.
                  A BSplineEdge must satisfy eval(0) ≈ vertices[i] and
                  eval(1) ≈ vertices[(i+1)%N].
    t           : sheet thickness
    name        : label string
    corner_r    : rounding radius at free convex corners (straight-edge corners only)
    """

    def __init__(self, origin, T, W, vertices, edge_curves=None,
                 t=2.0, name='SP', corner_r=0.0):
        self.origin   = np.asarray(origin, float)
        self.T        = _n(np.asarray(T, float))
        self.W        = _n(np.asarray(W, float))
        verts = np.asarray(vertices, float)
        if _signed_area2(verts) < 0:
            verts = verts[::-1]
        self.vertices = verts
        self.t        = float(t)
        self.name     = name
        self.corner_r = float(corner_r)
        n = len(verts)
        if edge_curves is None:
            self._edge_curves = [None] * n
        else:
            if len(edge_curves) != n:
                raise ValueError(
                    f'edge_curves must have {n} entries (one per edge), got {len(edge_curves)}.')
            self._edge_curves = list(edge_curves)
        self._edge_spans = {}

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def N(self):
        return _n(np.cross(self.T, self.W))

    def _pt3(self, u, v):
        return self.origin + float(u)*self.T + float(v)*self.W

    @property
    def num_edges(self):
        return len(self.vertices)

    def edge_curve(self, i):
        """Return the BSplineEdge for edge i, or None if the edge is straight."""
        return self._edge_curves[i % self.num_edges]

    # ── child tracking (identical logic to PolygonPanel) ──────────────────────

    def _register_child(self, edge_idx, start_frac=0.0, end_frac=1.0):
        sf, ef = float(start_frac), float(end_frac)
        spans  = self._edge_spans.setdefault(edge_idx, [])
        for s, e in spans:
            if sf < e - 1e-6 and ef > s + 1e-6:
                raise ValueError(
                    f'Child [{sf:.3g},{ef:.3g}] overlaps existing child '
                    f'[{s:.3g},{e:.3g}] on edge {edge_idx}.')
        spans.append((sf, ef))

    def _corner_blocked(self, i):
        n = self.num_edges
        for sf, ef in self._edge_spans.get((i - 1) % n, []):
            if ef > 1.0 - 1e-6:
                return True
        for sf, ef in self._edge_spans.get(i, []):
            if sf < 1e-6:
                return True
        return False

    def _corner_trim(self, i):
        """Trim setback at vertex i, using curve tangent where the edge is curved."""
        if self.corner_r == 0.0 or self._corner_blocked(i):
            return 0.0
        n = self.num_edges
        # incoming tangent: end of edge (i-1)
        prev_crv = self._edge_curves[(i - 1) % n]
        if prev_crv is not None:
            tang = prev_crv.tangent(1.0)
            norm = np.linalg.norm(tang)
            if norm < 1e-12:
                return 0.0
            e_in = tang / norm
        else:
            raw = self.vertices[i] - self.vertices[(i - 1) % n]
            l   = np.linalg.norm(raw)
            if l < 1e-12:
                return 0.0
            e_in = raw / l
        # outgoing tangent: start of edge i
        curr_crv = self._edge_curves[i]
        if curr_crv is not None:
            tang = curr_crv.tangent(0.0)
            norm = np.linalg.norm(tang)
            if norm < 1e-12:
                return 0.0
            e_out = tang / norm
        else:
            raw = self.vertices[(i + 1) % n] - self.vertices[i]
            l   = np.linalg.norm(raw)
            if l < 1e-12:
                return 0.0
            e_out = raw / l
        cross = float(e_in[0]*e_out[1] - e_in[1]*e_out[0])
        if cross <= 1e-6:
            return 0.0
        dot   = float(np.dot(e_in, e_out))
        sin_h = np.sqrt(max(0.0, (1.0 - dot) / 2.0))
        cos_h = np.sqrt(max(0.0, (1.0 + dot) / 2.0))
        if cos_h < 1e-9:
            return 0.0
        return self.corner_r * sin_h / cos_h

    def _edge_flat_trims(self, edge_idx):
        if self.corner_r == 0.0:
            return 0.0, 0.0
        n = self.num_edges
        return self._corner_trim(edge_idx), self._corner_trim((edge_idx + 1) % n)

    # ── geometry ──────────────────────────────────────────────────────────────

    def edge(self, i):
        """Return (p0, p1, chord_dir, outward_dir) for edge i.

        Uses chord geometry for compatibility with extend_panel().
        Curved geometry is accessible via edge_curve(i).
        """
        n      = self.num_edges
        u0, v0 = self.vertices[i % n]
        u1, v1 = self.vertices[(i + 1) % n]
        p0 = self._pt3(u0, v0)
        p1 = self._pt3(u1, v1)
        du, dv = u1 - u0, v1 - v0
        L = np.hypot(du, dv)
        if L < 1e-12:
            raise ValueError(f'SplinePanel edge {i} has zero chord length.')
        ed  = (du/L)*self.T + (dv/L)*self.W
        out = (dv/L)*self.T + (-du/L)*self.W
        return p0.copy(), p1.copy(), _n(ed), _n(out)

    def faces(self, n_curve=12):
        """Triangulate the panel face.

        Curved edges are sampled at n_curve interior points plus each endpoint.
        Straight edges contribute only their start vertex; the end vertex is the
        start of the following edge.  The resulting closed polygon is triangulated
        via ear-clipping.

        When corner_r > 0 and ALL edges are straight, the existing
        _rounded_polygon_pts path is used (identical to PolygonPanel).
        When any edge is curved, corner rounding is omitted for that junction
        (the B-spline curve already provides smooth geometry).
        """
        n       = self.num_edges
        any_crv = any(c is not None for c in self._edge_curves)

        if not any_crv:
            # All-straight: delegate to same logic as PolygonPanel
            if self.corner_r > 0.0:
                radii  = np.array([
                    0.0 if self._corner_blocked(i) else self.corner_r
                    for i in range(n)
                ])
                pts_2d = _rounded_polygon_pts(self.vertices, radii)
                pts_3d = [self._pt3(u, v) for u, v in pts_2d]
                tris   = _ear_clip(np.array(pts_2d))
                return [[pts_3d[i], pts_3d[j], pts_3d[k]] for i, j, k in tris]
            tris = _ear_clip(self.vertices)
            return [[self._pt3(*self.vertices[i]),
                     self._pt3(*self.vertices[j]),
                     self._pt3(*self.vertices[k])]
                    for i, j, k in tris]

        # Mixed or all-curved: build boundary polygon by sampling each edge
        pts_2d = []
        for i in range(n):
            crv = self._edge_curves[i]
            if crv is not None:
                # n_curve+1 samples t=0..1; exclude last (== next vertex)
                seg = crv.sample(n_curve + 1)[:-1]
                pts_2d.extend(seg.tolist())
            else:
                pts_2d.append(self.vertices[i].tolist())
        pts_2d = np.array(pts_2d, float)
        tris   = _ear_clip(pts_2d)
        return [[self._pt3(*pts_2d[i]), self._pt3(*pts_2d[j]), self._pt3(*pts_2d[k])]
                for i, j, k in tris]


# ── CurvedBendArc ─────────────────────────────────────────────────────────────

class CurvedBendArc:
    """Cylindrical arc strip swept along a curved (B-spline) parent edge.

    At each sample point along the edge curve, the local outward tangent is used
    to orient the arc, producing a swept surface whose axis follows the curve.
    For a straight edge this reduces to BendArc.

    Parameters
    ----------
    edge_curve   : BSplineEdge in the parent panel's (u, v) local coordinates
    panel_origin : 3-D origin of the parent panel
    panel_T      : T basis vector of the parent panel
    panel_W      : W basis vector of the parent panel
    N_bend       : bend direction (signed panel normal), 3-D unit vector
    alpha        : total bend angle in radians
    r_n          : neutral-axis bend radius (inner_r + t/2)
    n_edge       : number of arc-length samples along the edge (default 32)
    n_arc        : number of arc angle subdivisions (default 28)
    """

    def __init__(self, edge_curve, panel_origin, panel_T, panel_W,
                 N_bend, alpha, r_n, n_edge=32, n_arc=28):
        self.edge_curve   = edge_curve
        self.panel_origin = np.asarray(panel_origin, float)
        self.panel_T      = np.asarray(panel_T, float)
        self.panel_W      = np.asarray(panel_W, float)
        self.N_bend       = np.asarray(N_bend, float)
        self.alpha        = float(alpha)
        self.r_n          = float(r_n)
        self.n_edge       = int(n_edge)
        self.n_arc        = int(n_arc)

    def _build_grid(self):
        """Compute the (n_arc+1) × (n_edge+1) point grid for the swept surface."""
        n_e  = self.n_edge
        n_a  = self.n_arc
        orig = self.panel_origin
        pT   = self.panel_T
        pW   = self.panel_W
        Nb   = self.N_bend
        r_n  = self.r_n

        # arc-length-spaced samples along the edge curve
        ts_al = np.linspace(0.0, 1.0, n_e + 1)  # uniform-t fallback
        pts_uv = self.edge_curve.sample_arclength(n_e + 1)   # (n_e+1, 2)
        tangs  = self.edge_curve.tangent(ts_al)               # (n_e+1, 2)

        # local outward normals in 3-D at each edge sample
        # T_out = rotate 2-D tangent 90° CW (outward for CCW polygon) → in 3-D panel plane
        T_outs = np.array([
            _n(tang[1]*pT - tang[0]*pW)
            for tang in tangs
        ])  # (n_e+1, 3)

        # 3-D edge points
        p_edges = np.array([
            orig + uv[0]*pT + uv[1]*pW
            for uv in pts_uv
        ])  # (n_e+1, 3)

        # build grid: grid[a_idx, k] = 3-D point
        alphas = np.linspace(0.0, self.alpha, n_a + 1)
        grid   = np.empty((n_a + 1, n_e + 1, 3))
        for ai, a in enumerate(alphas):
            ca, sa = np.cos(a), np.sin(a)
            for k in range(n_e + 1):
                C = p_edges[k] + r_n * Nb
                grid[ai, k] = C + r_n * (-ca * Nb + sa * T_outs[k])
        return grid

    def faces(self):
        """Return list of quads (n_arc × n_edge) tessellating the swept surface."""
        grid = self._build_grid()
        n_a, n_e = self.n_arc, self.n_edge
        quads = []
        for ai in range(n_a):
            for k in range(n_e):
                quads.append([
                    grid[ai,     k    ],
                    grid[ai,     k + 1],
                    grid[ai + 1, k + 1],
                    grid[ai + 1, k    ],
                ])
        return quads
