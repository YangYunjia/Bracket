"""
panels.py — panel types for sheet metal brackets.

"""

import numpy as np
from .maths import n, signed_area2
from .curve import Curve3D

from typing import Union

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

    def __init__(self, name, origin, T, W, vertices, corner_r=0.0):
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


    def __init__(self, name, origin, T, W, length, width,
                 corner_r=0.0):

        self.length = float(length)
        self.width = float(width)
        vertices = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]])
        super().__init__(name, origin, T, W, vertices, corner_r)


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

    def __init__(self, name, profiles, t_start=None, t_end=None,
                 spine_pts=None, ruled=False):
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

    def __init__(self, name, profiles, t_start=None, t_end=None,
                 spine_pts=None, ruled=False):
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


Panel = Union[FlatPanel, LoftPanel, PipePanel]