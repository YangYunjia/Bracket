
'''
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
        self.T        = n(np.asarray(T, float))
        self.W        = n(np.asarray(W, float))
        verts = np.asarray(vertices, float)
        if signed_area2(verts) < 0:
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
        return n(np.cross(self.T, self.W))

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
        return p0.copy(), p1.copy(), n(ed), n(out)

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
            n(tang[1]*pT - tang[0]*pW)
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


'''