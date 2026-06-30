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

    def __init__(self, origin, T, W, vertices, name='P', corner_r=0.0):
        """
        Parameters
        ----------
        origin   : 3-D anchor point
        T, W     : two orthogonal 3-D vectors spanning the panel plane
        vertices : (N, 2) array-like of (u, v) pairs, N >= 3
        name     : label string
        corner_r : rounding radius for free corners
        """
        self.origin      = np.asarray(origin, float)
        self.T           = n(T)
        self.W           = n(W)
        self.name        = str(name)
        self.corner_r    = float(corner_r)

        pts = np.asarray(vertices, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError('vertices must be (N>=3, 2) array of (u, v) pairs.')
        if signed_area2(pts) < 0:
            pts = pts[::-1]        # normalise to CCW
        self.vertices = pts        # (N, 2), relative to the origin

        self._edge_lengths = [np.linalg.norm(self.start_idx(i) - self.end_idx(i)) for i in range(len(vertices))]
        self._edge_spans = [[] for _ in range(len(vertices))] # edge_idx → [(start_frac, end_frac), ...] (indicate the occupied portion of the edge)
        # self.corner_blocked = [False] * len(vertices) # indicate if the corner is blocked by a child panel
        self.corner_trim    = [self._initial_corner_trim(i) for i in range(len(vertices))] # compute the corner-rounding trim length at each vertex

    @property
    def N(self):
        return n(np.cross(self.T, self.W))

    def _pt3(self, u, v):
        return self.origin + float(u)*self.T + float(v)*self.W

    @property
    def num_edges(self):
        return len(self.vertices)

    def edge_length(self, edge_idx):
        return self._edge_lengths[edge_idx]

    def start_idx(self, edge_idx):
        return edge_idx

    def end_idx(self, edge_idx):
        return (edge_idx + 1) % len(self.vertices)

    def is_edge_occupied(self, edge_idx: int, start_frac: float, end_frac: float) -> bool:
        return any(float(start_frac) < e - 1e-6 and float(end_frac) > s + 1e-6 for s, e in self._edge_spans[edge_idx])

    def _register_child(self, edge_idx: int, start_frac: float = 0.0, end_frac: float = 1.0):
        """Record a child panel spanning [start_frac, end_frac] on this edge.
        """
        self._edge_spans[edge_idx].append((float(start_frac), float(end_frac)))

    def _initial_corner_trim(self, i: int) -> float:
        """Compute the corner-rounding trim length at vertex i.

        Returns 0.0 if corner_r == 0, the vertex is concave, or the corner is
        blocked by a registered child.  Otherwise uses the half-angle formula:
            trim = corner_r * tan(φ/2)
        where φ is the exterior deflection angle at vertex i.
        """
        if self.corner_r == 0.0:
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
        return self.corner_trim[self.start_idx(edge_idx)], self.corner_trim[self.end_idx(edge_idx)]

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for polygon edge i.

        Compatible with extend_panel / SheetMetalModel.extend().
        """
        verts = self.vertices
        dvert = (verts[self.end_idx(i)] - verts[self.start_idx(i)])
        p0 = self._pt3(*verts[self.start_idx(i)])
        p1 = self._pt3(*verts[self.end_idx(i)])

        # Edge direction in 3-D
        ed  = dvert[0]*self.T + dvert[1]*self.W
        # Outward normal: right-hand perp of CCW edge = rotate −90°
        out = dvert[1] * self.T - dvert[0] * self.W

        return p0.copy(), p1.copy(), n(ed), n(out)


class FlatPanel(PolygonPanel):
    """Flat rectangular panel (mid-surface only).
    The panel lives in the plane spanned by T and W from origin.

    attributes:
        origin : 3-D anchor point
        T, W   : two orthogonal 3-D vectors spanning the panel plane
        length : length of the panel in the t-direction
        width  : width of the panel in the w-direction
        t      : sheet thickness
        name   : label string
        corner_r : rounding radius for free corners
    """


    def __init__(self, origin, T, W, length, width, name='F',
                 corner_r=0.0):

        vertices = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]])
        super().__init__(origin, T, W, vertices, name, corner_r)


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

    profiles[0]  — root profile  (attaches to parent via BendArc)
    profiles[-1] — free profile  (children extend from here)
    profiles[1:-1] — optional intermediate cross-sections

    ruled=True  → straight rulings between corresponding profile points
    ruled=False → smooth skinned surface through all profiles

    Edge numbering (CCW virtual quad — same convention as PolygonPanel)
    ─────────────────────────────────────────────────────────────────
    Corner ring: v0=a0_r, v1=b0_r, v2=b0_f, v3=a0_f
      (profile endpoints: root a0_r→b0_r, free a0_f→b0_f at t=0→1)

      0 : root profile   v0 → v1
      1 : long edge      v1 → v2  (trailing / t=1 side)
      2 : free profile   v2 → v3  (extend children from here)
      3 : long edge      v3 → v0  (leading / t=0 side)

    extend() / pre_extend() support edges 0 and 2 only (the two profiles).
    """

    _EXTENDABLE_EDGES = (0, 2)

    def __init__(self, profiles, t_start=None, t_end=None,
                 spine_pts=None, ruled=False, name='LP'):
        """
        Parameters
        ----------
        profiles  : list of Curve3D (or [(x,y,z),(x,y,z)] shorthand)
        t_start   : optional unit tangent at the root edge (for G1 with BendArc)
        t_end     : optional unit tangent at the free edge (for G1 with BendArc)
        spine_pts : optional list of 3-D spine waypoints (one per profile);
                    defaults to profile midpoints
        ruled     : bool — ruled (True) or smooth (False) loft
        name      : label string
        """
        if len(profiles) < 2:
            raise ValueError('LoftPanel needs at least 2 profiles.')
        def _coerce(p):
            if isinstance(p, Curve3D):
                return p
            pts = np.asarray(p, float)
            if pts.ndim == 1:
                raise ValueError('Profile must be a Curve3D or list of 3-D points.')
            if len(pts) == 2:
                return Curve3D.line(pts[0], pts[1])
            return Curve3D('polyline', _pts=pts)
        self.profiles    = [_coerce(p) for p in profiles]
        self.ruled       = bool(ruled)
        self.name        = str(name)
        self.corner_r    = 0.0
        self.corner_trim = [0.0, 0.0, 0.0, 0.0]
        self._edge_spans = [[] for _ in range(4)]

        self.t_start = n(np.asarray(t_start, float)) if t_start is not None else None
        self.t_end   = n(np.asarray(t_end,   float)) if t_end   is not None else None

        if spine_pts is not None:
            if len(spine_pts) != len(self.profiles):
                raise ValueError('spine_pts must have the same count as profiles.')
            self.spine_pts = [np.asarray(p, float) for p in spine_pts]
        else:
            self.spine_pts = [(a + b) / 2.0
                              for a, b in [p.endpoints() for p in self.profiles]]

    # ── CCW corner ring (PolygonPanel-compatible indexing) ───────────────────

    def _corners(self):
        """CCW corners v0..v3 of the loft boundary quad."""
        a0_r, b0_r = self.profiles[0].endpoints()
        a0_f, b0_f = self.profiles[-1].endpoints()
        return [np.asarray(p, float) for p in (a0_r, b0_r, b0_f, a0_f)]

    @property
    def num_edges(self):
        return 4

    def start_idx(self, edge_idx):
        return int(edge_idx)

    def end_idx(self, edge_idx):
        return (int(edge_idx) + 1) % 4

    def edge_length(self, edge_idx):
        c = self._corners()
        i = int(edge_idx)
        return float(np.linalg.norm(c[(i + 1) % 4] - c[i]))

    def is_edge_occupied(self, edge_idx: int, start_frac: float, end_frac: float) -> bool:
        return any(float(start_frac) < e - 1e-6 and float(end_frac) > s + 1e-6
                   for s, e in self._edge_spans[edge_idx])

    def _register_child(self, edge_idx: int, start_frac: float = 0.0, end_frac: float = 1.0):
        self._edge_spans[edge_idx].append((float(start_frac), float(end_frac)))

    # ── normal helpers ────────────────────────────────────────────────────────

    def _edge_normal(self, profile_idx, long_side=0):
        """Approximate panel normal at a given profile.

        Computed as cross(profile_tangent_at_t=long_side_end, long_edge_dir).
        long_side=0 → use first endpoint; long_side=1 → use second endpoint.
        """
        prof = self.profiles[profile_idx]
        t_end = 0.0 if long_side == 0 else 1.0
        tang_prof = prof.tangent_at(t_end)

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
        """Panel normal at boundary edge edge_idx (extend / bend protocol)."""
        mid = len(self.profiles) // 2
        if edge_idx == 0:
            return self._edge_normal(0, long_side=0)
        if edge_idx == 1:
            return self._edge_normal(mid, long_side=1)
        if edge_idx == 2:
            return self._edge_normal(-1, long_side=1)
        if edge_idx == 3:
            return self._edge_normal(mid, long_side=0)
        raise IndexError(f'LoftPanel has 4 edges (0–3), got {edge_idx}.')

    @property
    def N(self):
        return self.N_at_edge(0)

    # ── edge geometry (PolygonPanel-compatible) ─────────────────────────────

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for CCW edge i.

        outward_dir matches PolygonPanel: in-plane exterior normal
        T_out = normalise(edge_dir × N).
        """
        if i not in (0, 1, 2, 3):
            raise IndexError(f'LoftPanel has 4 edges (0–3), got {i}.')
        corners = self._corners()
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        ed = n(p1 - p0)
        out = n(np.cross(ed, self.N_at_edge(i)))
        return p0.copy(), p1.copy(), ed, out


# ── PipePanel ──────────────────────────────────────────────────────────────────

class PipePanel:
    """Panel swept along a spline spine with controlled start/end tangents.

    Parallel to LoftPanel but rendered via BRepOffsetAPI_MakePipeShell in the
    OCC backend, guaranteeing G1 continuity with adjacent BendArcs.

    The spine is an interpolating B-spline through ``spine_pts`` (one per
    profile).  If ``spine_pts`` is omitted it defaults to the midpoints of the
    profile endpoints.  ``t_start`` / ``t_end`` are optional unit tangent
    constraints at the first / last spine point.

    Edge numbering: identical to LoftPanel (0=root, 1=free, 2/3=long sides).
    """

    def __init__(self, profiles, t_start=None, t_end=None,
                 spine_pts=None, ruled=False, name='PP', t=2.0):
        if len(profiles) < 2:
            raise ValueError('PipePanel needs at least 2 profiles.')

        def _coerce(p):
            if isinstance(p, Curve3D):
                return p
            pts = np.asarray(p, float)
            if pts.ndim == 1:
                raise ValueError('Profile must be a Curve3D or list of 3-D points.')
            if len(pts) == 2:
                return Curve3D.line(pts[0], pts[1])
            return Curve3D('polyline', _pts=pts)

        self.profiles    = [_coerce(p) for p in profiles]
        self.ruled       = bool(ruled)
        self.name        = str(name)
        self.t           = float(t)
        self._edge_spans = {}

        self.t_start = n(np.asarray(t_start, float)) if t_start is not None else None
        self.t_end   = n(np.asarray(t_end,   float)) if t_end   is not None else None

        if spine_pts is not None:
            if len(spine_pts) != len(self.profiles):
                raise ValueError('spine_pts must have the same count as profiles.')
            self.spine_pts = [np.asarray(p, float) for p in spine_pts]
        else:
            self.spine_pts = [(a + b) / 2.0
                              for a, b in [p.endpoints() for p in self.profiles]]

    # ── normal helpers (identical to LoftPanel) ───────────────────────────────

    def _edge_normal(self, profile_idx, long_side=0):
        prof     = self.profiles[profile_idx]
        t_ep     = 0.0 if long_side == 0 else 1.0
        tang     = prof.tangent_at(t_ep)
        a0, _    = self.profiles[0].endpoints()
        a1, _    = self.profiles[-1].endpoints()
        _, b0    = self.profiles[0].endpoints()
        _, b1    = self.profiles[-1].endpoints()
        long_dir = n(a1 - a0) if long_side == 0 else n(b1 - b0)
        return n(np.cross(tang, long_dir))

    def N_at_edge(self, edge_idx):
        if edge_idx == 0:
            return self._edge_normal(0,  long_side=0)
        if edge_idx == 1:
            return self._edge_normal(-1, long_side=0)
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
                    f'[{s:.3g},{e:.3g}] on PipePanel edge {edge_idx}.')
        spans.append((sf, ef))

    def _corner_blocked(self, corner_idx):
        return False

    # ── edge geometry (extend_panel / extend compatible) ─────────────────────

    def edge(self, i):
        """Return (p0, p1, edge_dir, outward_dir) for edge i.

        At edge 1 (free profile) the outward direction uses ``t_end`` when it
        is set, so the downstream BendArc is built tangentially consistent with
        the constrained spine end.
        """
        a0_r, b0_r = self.profiles[0].endpoints()
        a0_f, b0_f = self.profiles[-1].endpoints()

        if i == 0:   # root profile
            chord = n(b0_r - a0_r)
            out   = n(np.cross(chord, self.N_at_edge(0)))
            return a0_r.copy(), b0_r.copy(), chord, out
        if i == 1:   # free profile
            chord = n(b0_f - a0_f)
            out   = self.t_end.copy() if self.t_end is not None else n(a0_f - a0_r)
            return a0_f.copy(), b0_f.copy(), chord, out
        if i == 2:
            return a0_r.copy(), a0_f.copy(), n(a0_f - a0_r), self.N_at_edge(2)
        if i == 3:
            return b0_r.copy(), b0_f.copy(), n(b0_f - b0_r), self.N_at_edge(3)
        raise IndexError(f'PipePanel has 4 edges (0–3), got {i}.')
