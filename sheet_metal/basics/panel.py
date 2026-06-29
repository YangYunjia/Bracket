"""
sheet_metal.primitives — geometry primitives for mid-surface shell models.

FlatPanel      : flat rectangular panel
BendArc        : full-edge cylindrical arc strip
PartialBendArc : partial-edge arc with smooth width taper
PolygonPanel   : flat panel with arbitrary polygon outline
# BSplineEdge    : cubic B-spline curve for one panel edge
# SplinePanel    : flat panel with curved (B-spline) edge outlines
# CurvedBendArc  : arc strip that sweeps along a curved (B-spline) edge
Curve3D        : 3-D world-space curve (line / arc / B-spline)
LoftPanel      : panel defined by 2+ Curve3D profiles (ruled or smooth loft)
"""

import numpy as np
from .maths import n, signed_area2


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
        self.T           = n(T)
        self.W           = n(W)
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
        return n(np.cross(self.T, self.W))

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
        return p0.copy(), p1.copy(), n(ed), n(out)


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
        self.T           = n(T)
        self.W           = n(W)
        self.t           = float(t)
        self.name        = str(name)
        self.corner_r    = float(corner_r)
        self._edge_spans = {}

        pts = np.asarray(vertices, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError('vertices must be (N>=3, 2) array of (u, v) pairs.')
        if signed_area2(pts) < 0:
            pts = pts[::-1]        # normalise to CCW
        self.vertices = pts        # (N, 2)

    @property
    def N(self):
        return n(np.cross(self.T, self.W))

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
        return p0.copy(), p1.copy(), n(ed), n(out)


# ── Curve3D ───────────────────────────────────────────────────────────────────

class Curve3D:
    """Lightweight 3-D world-space curve.

    Three constructors:
        Curve3D.line(p0, p1)
        Curve3D.arc(center, axis, radius, p_start, p_end)
        Curve3D.bspline(control_pts, degree=3)

    All expose:
        sample(n)       → (n, 3) float array, arc-length-uniform
        tangent_at(t)   → unit tangent at parameter t ∈ [0, 1]
        endpoints()     → (p0, p1) 3-D arrays (t=0, t=1 points)
        arc_length()    → float

    The `_kind` attribute is read by the OCC builder to produce exact geometry
    instead of a polygon approximation.
    """

    def __init__(self, _kind, **kw):
        self._kind = _kind
        self.__dict__.update(kw)

    # ── constructors ─────────────────────────────────────────────────────────

    @staticmethod
    def line(p0, p1):
        """Straight line segment from p0 to p1."""
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        return Curve3D('line', p0=p0, p1=p1)

    @staticmethod
    def arc(center, axis, radius, p_start, p_end):
        """Circular arc.

        Parameters
        ----------
        center  : 3-D centre of the circle
        axis    : normal to the circle plane (CCW positive)
        radius  : circle radius
        p_start : 3-D point on the arc (t=0); must lie on the circle
        p_end   : 3-D point on the arc (t=1); must lie on the circle
        """
        center  = np.asarray(center,  float)
        axis    = n(np.asarray(axis, float))
        p_start = np.asarray(p_start, float)
        p_end   = np.asarray(p_end,   float)
        return Curve3D('arc', center=center, axis=axis,
                       radius=float(radius),
                       p_start=p_start, p_end=p_end)

    @staticmethod
    def bspline(control_pts, degree=3):
        """3-D B-spline curve through (or near) control_pts.

        Parameters
        ----------
        control_pts : (M, 3) control polygon in world coordinates
        degree      : spline degree (default 3 = cubic)
        """
        from scipy.interpolate import BSpline as _BSpline
        cp = np.asarray(control_pts, float)
        if cp.ndim != 2 or cp.shape[1] != 3:
            raise ValueError('control_pts must be (M, 3).')
        m   = len(cp)
        deg = int(degree)
        if m < deg + 1:
            raise ValueError(f'Need at least {deg+1} control points for degree {deg}.')
        n_int  = m - deg - 1
        intr   = (np.linspace(0.0, 1.0, n_int + 2)[1:-1]
                  if n_int > 0 else np.empty(0))
        knots  = np.concatenate([np.zeros(deg+1), intr, np.ones(deg+1)])
        sps    = [_BSpline(knots, cp[:, d], deg) for d in range(3)]
        dsps   = [sp.derivative() for sp in sps]
        return Curve3D('bspline', _cp=cp, _deg=deg, _knots=knots,
                       _sps=sps, _dsps=dsps)

    # ── evaluation ───────────────────────────────────────────────────────────

    def _eval_raw(self, t):
        """Evaluate curve at parameter t (scalar or array), no arc-length remapping."""
        t = np.asarray(t, float)
        if self._kind == 'line':
            return self.p0 + t[..., np.newaxis] * (self.p1 - self.p0)
        if self._kind == 'arc':
            return self._arc_eval(t)
        # bspline
        return np.stack([sp(t) for sp in self._sps], axis=-1)

    def _arc_eval(self, t):
        """Evaluate arc at scalar or array t."""
        c   = self.center
        ax  = self.axis
        r   = self.radius
        # Build local frame in the circle plane
        v0  = self.p_start - c
        # Ensure v0 is perpendicular to axis (project out axis component)
        v0  = v0 - np.dot(v0, ax) * ax
        v0  = r * n(v0)
        v1  = np.cross(ax, v0)        # 90° CCW in the plane
        # angle from p_start to p_end CCW around axis
        ve  = self.p_end - c
        ve  = ve - np.dot(ve, ax) * ax
        ve  = n(ve)
        cos_a = np.clip(np.dot(n(v0), ve), -1.0, 1.0)
        sin_a = np.dot(np.cross(n(v0), ve), ax)
        total = np.arctan2(sin_a, cos_a)
        if total < 0:
            total += 2 * np.pi
        t = np.asarray(t, float)
        angle = t * total
        return c + r * (np.cos(angle)[..., np.newaxis] * n(v0)
                        + np.sin(angle)[..., np.newaxis] * n(v1))

    def _arc_length_table(self, n=512):
        ts  = np.linspace(0.0, 1.0, n + 1)
        pts = self._eval_raw(ts)
        cum = np.concatenate([[0.0],
                              np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        return ts, cum

    def arc_length(self, n=512):
        _, cum = self._arc_length_table(n)
        return float(cum[-1])

    def sample(self, n, uniform=True):
        """Return n 3-D points.

        uniform=True  : evenly spaced in arc length (default)
        uniform=False : evenly spaced in parameter t
        """
        if not uniform or self._kind == 'line':
            ts = np.linspace(0.0, 1.0, n)
            return self._eval_raw(ts)
        ts_raw, cum = self._arc_length_table(max(512, n * 8))
        total = cum[-1]
        ts_out = np.interp(np.linspace(0.0, total, n), cum, ts_raw)
        return self._eval_raw(ts_out)

    def tangent_at(self, t):
        """Unit tangent at parameter t ∈ [0, 1] (scalar)."""
        t = float(t)
        if self._kind == 'line':
            return n(self.p1 - self.p0)
        if self._kind == 'arc':
            eps = 1e-5
            t1  = min(t + eps, 1.0)
            t0  = max(t - eps, 0.0)
            return n(self._arc_eval(np.array([t1]))[0]
                      - self._arc_eval(np.array([t0]))[0])
        # bspline
        d = np.array([dsp(t) for dsp in self._dsps])
        return n(d)

    def endpoints(self):
        """Return (p0, p1) — the t=0 and t=1 world-space points."""
        if self._kind == 'line':
            return self.p0.copy(), self.p1.copy()
        pts = self._eval_raw(np.array([0.0, 1.0]))
        return pts[0].copy(), pts[1].copy()


# ── LoftPanel ─────────────────────────────────────────────────────────────────

class LoftPanel:
    """Panel defined by 2 or more Curve3D profiles (ruled or smooth loft).

    profiles[0]  — root edge  (attaches to parent via BendArc)
    profiles[-1] — free edge  (children extend from here)
    profiles[1:-1] — optional intermediate cross-sections

    ruled=True  → straight rulings between corresponding profile points
                  (ThruSections ruled surface in OCC)
    ruled=False → smooth skinned surface through all profiles

    Edge numbering
    ──────────────
      0 : root profile  (profiles[0])
      1 : free profile  (profiles[-1])
      2 : "first" long edge  profiles[0].endpoints()[0] → profiles[-1].endpoints()[0]
      3 : "second" long edge profiles[0].endpoints()[1] → profiles[-1].endpoints()[1]

    N_at_edge(i) returns the panel normal at that edge, needed by extend_panel()
    when child panels are grown from this LoftPanel.
    """

    def __init__(self, profiles, ruled=True, name='LP', t=2.0):
        """
        Parameters
        ----------
        profiles : list of Curve3D (or list of (x,y,z) tuples → converted to
                   straight-line Curve3D automatically if exactly 2 entries)
        ruled    : bool — ruled (True) or smooth (False) loft
        name     : label string
        t        : sheet thickness
        """
        if len(profiles) < 2:
            raise ValueError('LoftPanel needs at least 2 profiles.')
        # Auto-convert bare point lists: [(x,y,z),(x,y,z)] → Curve3D.line
        def _coerce(p):
            if isinstance(p, Curve3D):
                return p
            pts = np.asarray(p, float)
            if pts.ndim == 1:
                raise ValueError('Profile must be a Curve3D or list of 3-D points.')
            if len(pts) == 2:
                return Curve3D.line(pts[0], pts[1])
            # >2 points: treat as polyline; build a degree-1 bspline approximation
            # (just store as dense samples for now — OCC builder will polygon-wire them)
            return Curve3D('polyline', _pts=pts)
        self.profiles  = [_coerce(p) for p in profiles]
        self.ruled     = bool(ruled)
        self.name      = str(name)
        self.t         = float(t)
        self._edge_spans = {}   # same child-tracking interface as FlatPanel

    # ── normal helpers ────────────────────────────────────────────────────────

    def _edge_normal(self, profile_idx, long_side=0):
        """Approximate panel normal at a given profile.

        Computed as cross(profile_tangent_at_t=long_side_end, long_edge_dir).
        long_side=0 → use first endpoint; long_side=1 → use second endpoint.
        """
        prof = self.profiles[profile_idx]
        t_end = 0.0 if long_side == 0 else 1.0
        tang_prof = prof.tangent_at(t_end)

        # Long-edge direction: connecting same end across profiles
        a0, _ = self.profiles[0].endpoints()
        a1, _ = self.profiles[-1].endpoints()
        _, b0 = self.profiles[0].endpoints()
        _, b1 = self.profiles[-1].endpoints()
        if long_side == 0:
            long_dir = n(a1 - a0)
        else:
            long_dir = n(b1 - b0)
        return n(np.cross(tang_prof, long_dir))

    def N_at_edge(self, edge_idx):
        """Panel normal at a boundary edge (needed by extend_panel protocol)."""
        if edge_idx == 0:
            return self._edge_normal(0, long_side=0)
        if edge_idx == 1:
            return self._edge_normal(-1, long_side=0)
        # long edges 2/3: midpoint normal (rare to extend from long edges)
        return self._edge_normal(len(self.profiles) // 2, long_side=(edge_idx - 2))

    @property
    def N(self):
        return self.N_at_edge(0)

    # ── child tracking ────────────────────────────────────────────────────────

    def _register_child(self, edge_idx, start_frac=0.0, end_frac=1.0):
        sf, ef = float(start_frac), float(end_frac)
        spans  = self._edge_spans.setdefault(edge_idx, [])
        for s, e in spans:
            if sf < e - 1e-6 and ef > s + 1e-6:
                raise ValueError(
                    f'Child [{sf:.3g},{ef:.3g}] overlaps existing child '
                    f'[{s:.3g},{e:.3g}] on LoftPanel edge {edge_idx}.')
        spans.append((sf, ef))

    def _corner_blocked(self, corner_idx):
        return False   # LoftPanel corners are never rounded

    # ── edge geometry (extend_panel compatible) ───────────────────────────────

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for edge i.

        For edges 0 and 1 (profile edges) p0/p1 are the profile endpoints;
        edge_dir is the chord direction; outward_dir is the long-edge direction
        (pointing away from the opposite profile).
        For edges 2 and 3 (long edges) p0/p1 span from root to free profile.
        """
        a0_r, b0_r = self.profiles[0].endpoints()   # root profile endpoints
        a0_f, b0_f = self.profiles[-1].endpoints()  # free profile endpoints

        if i == 0:   # root profile
            chord = n(b0_r - a0_r)
            long  = n(a0_r - a0_f)   # pointing away from free end
            out   = n(np.cross(chord, self.N_at_edge(0)))
            return a0_r.copy(), b0_r.copy(), chord, out
        if i == 1:   # free profile — outward dir points away from root
            chord = n(b0_f - a0_f)
            out   = n(a0_f - a0_r)
            return a0_f.copy(), b0_f.copy(), chord, out
        if i == 2:   # first long edge (t=0 side)
            return a0_r.copy(), a0_f.copy(), n(a0_f - a0_r), self.N_at_edge(2)
        if i == 3:   # second long edge (t=1 side)
            return b0_r.copy(), b0_f.copy(), n(b0_f - b0_r), self.N_at_edge(3)
        raise IndexError(f'LoftPanel has 4 edges (0–3), got {i}.')

