"""
sheet_metal.recess — stamped/drawn recess feature.

Geometry (cross-section, outside → inside):
  panel surface → top bend (r_top) → straight wall (wall_h)
                → bottom bend (r_bot) → flat bottom

Without draft:  depth = r_top + wall_h + r_bot
With draft α:   depth = (r_top + r_bot)·(1 − sin α) + wall_h
  where α = draft_angle in radians (0 = vertical wall).
  The wall leans inward at angle α from the panel normal.
"""

import numpy as np
from .primitives import _n


# ── boundary helpers ──────────────────────────────────────────────────────────

def _boundary_segments(half_a, half_b, R_c):
    """Return list of (fn, arc_length) for the CCW rounded-rect boundary.

    fn(t): t ∈ [0,1] → (x, y, nx, ny)
           (x,y) = 2-D point,  (nx,ny) = inward unit normal.
    """
    W2, H2, R = half_a, half_b, R_c
    segs = []

    def _line(x0, y0, x1, y1, nx, ny):
        L = np.hypot(x1 - x0, y1 - y0)
        if L < 1e-12:
            return
        def fn(t, x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny):
            return x0 + t*(x1-x0), y0 + t*(y1-y0), nx, ny
        segs.append((fn, L))

    def _arc(cx, cy, a0, a1):
        L = abs(a1 - a0) * R
        if L < 1e-12:
            return
        def fn(t, cx=cx, cy=cy, a0=a0, a1=a1):
            a = a0 + t*(a1 - a0)
            x, y = cx + R*np.cos(a), cy + R*np.sin(a)
            return x, y, -np.cos(a), -np.sin(a)   # inward = toward center
        segs.append((fn, L))

    # CCW traversal starting from bottom of right side
    _line( W2,   -(H2-R),  W2,    H2-R,  -1.0,  0.0)   # right ↑
    _arc ( W2-R,  H2-R,    0,            np.pi/2)        # TR corner
    _line( W2-R,  H2,    -(W2-R), H2,     0.0, -1.0)    # top ←
    _arc (-(W2-R), H2-R,  np.pi/2,        np.pi)         # TL corner
    _line(-W2,    H2-R,   -W2,  -(H2-R),  1.0,  0.0)    # left ↓
    _arc (-(W2-R),-(H2-R), np.pi,      3*np.pi/2)        # BL corner
    _line(-(W2-R),-H2,     W2-R, -H2,     0.0,  1.0)    # bottom →
    _arc ( W2-R, -(H2-R), 3*np.pi/2,  2*np.pi)           # BR corner

    return segs


def _sample_segs(segs, N):
    """Sample N+1 points from a list of (fn, arc_length) boundary segments.

    fn(t) → (x, y, nx, ny) with t in [0,1] and (nx, ny) an inward unit normal.
    pts[N] == pts[0]  (closed boundary).
    Returns pts (N+1, 2) and norms (N+1, 2).
    """
    lens   = np.array([s[1] for s in segs])
    cumlen = np.concatenate([[0.0], np.cumsum(lens)])
    total  = cumlen[-1]

    pts   = np.empty((N+1, 2))
    norms = np.empty((N+1, 2))

    for i in range(N+1):
        s = (i % N) * total / N
        k = int(np.searchsorted(cumlen[1:], s, side='right'))
        k = min(k, len(segs) - 1)
        fn, L = segs[k]
        t = (s - cumlen[k]) / L if L > 1e-12 else 0.0
        x, y, nx, ny = fn(np.clip(t, 0.0, 1.0))
        pts[i]   = [x, y]
        norms[i] = [nx, ny]

    return pts, norms


def _sample_boundary(half_a, half_b, R_c, N):
    """Sample N+1 points on the rounded-rect boundary (closed: pts[N]==pts[0]).

    Returns pts (N+1,2) and norms (N+1,2) — inward unit normals.
    """
    return _sample_segs(_boundary_segments(half_a, half_b, R_c), N)


# ── cross-section profile ────────────────────────────────────────────────────

def _sample_profile(r_top, wall_h, r_bot, draft_angle, N):
    """Sample N+1 (u, v) pairs along the cross-section profile.

    u : inward offset from outer footprint boundary (>= 0)
    v : depth below panel surface (<= 0)

    Sequence: top-bend arc → straight wall → bottom-bend arc.

    draft_angle : angle from panel-normal (vertical) direction, radians.
                  0 = vertical wall (default).  Positive → leans inward.
    """
    α   = float(draft_angle)
    β   = 0.5*np.pi - α          # bend sweep for top and bottom arcs
    sin_a, cos_a = np.sin(α), np.cos(α)

    L_top  = r_top * β
    L_wall = wall_h / max(cos_a, 1e-9)   # path length along sloped wall
    L_bot  = r_bot * β
    total  = L_top + L_wall + L_bot

    # End of top bend
    u_te = r_top * cos_a
    v_te = r_top * (sin_a - 1.0)

    # Start of bottom bend (= end of wall)
    u_we = u_te + wall_h * (sin_a / max(cos_a, 1e-9))
    v_we = v_te - wall_h

    # Centre of bottom bend circle
    cu = u_we + r_bot * cos_a
    cv = v_we + r_bot * sin_a

    us = np.empty(N+1)
    vs = np.empty(N+1)

    for i in range(N+1):
        s = i * total / N

        if s <= L_top + 1e-12:
            θ = β * (s / L_top) if L_top > 1e-12 else β
            θ = np.clip(θ, 0.0, β)
            us[i] = r_top * np.sin(θ)
            vs[i] = r_top * (np.cos(θ) - 1.0)

        elif s <= L_top + L_wall + 1e-12:
            t_w   = s - L_top               # path along wall
            us[i] = u_te + t_w * sin_a
            vs[i] = v_te - t_w * cos_a

        else:
            t_b = s - L_top - L_wall        # path along bottom arc
            θ   = β * (t_b / L_bot) if L_bot > 1e-12 else β
            θ   = np.clip(θ, 0.0, β)
            us[i] = cu - r_bot * np.cos(α + θ)
            vs[i] = cv - r_bot * np.sin(α + θ)

    return us, vs


# ── StampedRecess ─────────────────────────────────────────────────────────────

class StampedRecess:
    """Stamped / drawn recess on a flat panel.

    The recess footprint is a rounded rectangle in the panel's (T, W) plane.
    width == height and corner_r == width/2 gives a circular recess.

    Parameters
    ----------
    panel       : FlatPanel
    cx, cy      : center of recess in panel local coords (from panel.origin,
                  along panel.T and panel.W respectively)
    width       : footprint outer width  (along panel.T)
    height      : footprint outer height (along panel.W)
    corner_r    : corner radius; clamped to min(width,height)/2
    r_top       : top bend radius  (must be > 0)
    r_bot       : bottom bend radius (must be > 0)
    depth       : total recess depth (specify depth OR wall_h)
    wall_h      : straight wall vertical height (derived from depth if omitted)
    draft_angle : angle from vertical (panel-normal) in degrees; default 0.
                  Positive values slope the wall inward toward the recess centre.
    n_boundary  : tessellation resolution around boundary
    n_profile   : tessellation resolution along cross-section
    """

    def __init__(self, panel, cx, cy, width, height, corner_r,
                 r_top, r_bot, depth=None, wall_h=None,
                 draft_angle=0.0,
                 n_boundary=80, n_profile=20):

        if r_top <= 0 or r_bot <= 0:
            raise ValueError('r_top and r_bot must be positive.')

        draft_angle = float(draft_angle)
        if draft_angle < 0.0 or draft_angle >= 90.0:
            raise ValueError(
                f'draft_angle must be in [0, 90) degrees, got {draft_angle}.')
        α     = np.radians(draft_angle)
        sin_a = np.sin(α)
        cos_a = np.cos(α)

        # depth = (r_top + r_bot)*(1 - sin α) + wall_h
        bend_drop = (r_top + r_bot) * (1.0 - sin_a)

        # ── resolve depth / wall_h ────────────────────────────────────────
        if depth is not None and wall_h is not None:
            expected = bend_drop + wall_h
            if abs(depth - expected) > 1e-6:
                raise ValueError(
                    f'Inconsistent: depth={depth} != '
                    f'(r_top+r_bot)*(1-sin(α))+wall_h={expected:.6g}')
        elif depth is not None:
            wall_h = float(depth) - bend_drop
        elif wall_h is not None:
            depth = bend_drop + float(wall_h)
        else:
            raise ValueError('Specify either depth or wall_h.')

        # ── clamp corner radius ───────────────────────────────────────────
        corner_r = min(float(corner_r), min(width, height) / 2.0)

        # ── validate ──────────────────────────────────────────────────────
        if wall_h < -1e-9:
            raise ValueError(
                f'wall_h = depth - bend_drop = {wall_h:.4g} < 0. '
                'Increase depth or reduce bend radii / draft angle.')
        wall_h = max(wall_h, 0.0)

        # Total inward offset at the bottom of the profile
        tan_a = np.tan(α)
        inset = (r_top + r_bot) * cos_a + wall_h * tan_a
        if corner_r < inset - 1e-9:
            raise ValueError(
                f'corner_r ({corner_r:.4g}) < profile inset ({inset:.4g}): '
                'profile would self-intersect at corners. '
                'Increase corner_r, reduce bend radii, or reduce draft_angle.')
        if width / 2.0 <= inset or height / 2.0 <= inset:
            raise ValueError(
                f'Footprint ({width}×{height}) too small for inset={inset:.4g}: '
                f'need both half-dims > {inset:.4g}.')
        # panel bounds check
        if (cx - width/2  < -1e-6 or cx + width/2  > panel.length + 1e-6 or
            cy - height/2 < -1e-6 or cy + height/2 > panel.width  + 1e-6):
            raise ValueError(
                f'Recess footprint [{cx-width/2:.3g},{cx+width/2:.3g}] × '
                f'[{cy-height/2:.3g},{cy+height/2:.3g}] lies outside '
                f'panel ({panel.length}×{panel.width}).')

        self.panel       = panel
        self.cx          = float(cx)
        self.cy          = float(cy)
        self.half_a      = width  / 2.0
        self.half_b      = height / 2.0
        self.corner_r    = float(corner_r)
        self.r_top       = float(r_top)
        self.r_bot       = float(r_bot)
        self.wall_h      = float(wall_h)
        self.depth       = float(depth)
        self.draft_angle = α                 # stored as radians internally
        self.n_boundary  = int(n_boundary)
        self.n_profile   = int(n_profile)

    def faces(self):
        p     = self.panel
        T, W_ = p.T, p.W
        Np    = p.N                                  # panel outward normal
        C3    = p.origin + self.cx*T + self.cy*W_   # recess centre in 3D

        bpts, bnorms = _sample_boundary(
            self.half_a, self.half_b, self.corner_r, self.n_boundary)
        us, vs = _sample_profile(
            self.r_top, self.wall_h, self.r_bot, self.draft_angle, self.n_profile)

        NB, NP = self.n_boundary, self.n_profile

        # 3-D grid: (NB+1) rows (boundary) × (NP+1) cols (profile)
        # S = boundary_pt + u * inward_normal_3d + v * Np
        # v < 0 and Np points outward → v*Np moves the point into the panel ✓
        grid = np.empty((NB+1, NP+1, 3))
        for i in range(NB+1):
            bp = C3 + bpts[i, 0]*T + bpts[i, 1]*W_
            n3 = bnorms[i, 0]*T + bnorms[i, 1]*W_
            for j in range(NP+1):
                grid[i, j] = bp + us[j]*n3 + vs[j]*Np

        out = []

        # Swept surface quads
        for i in range(NB):
            for j in range(NP):
                out.append([grid[i, j],   grid[i+1, j],
                            grid[i+1, j+1], grid[i, j+1]])

        # Flat bottom — fan triangles from centre
        centre_bot = C3 + vs[NP]*Np
        for i in range(NB):
            out.append([centre_bot, grid[i, NP], grid[i+1, NP]])

        return out


# ── polygon boundary ─────────────────────────────────────────────────────────

def _polygon_boundary_segments(vertices, corner_r=0.0):
    """Boundary segments for an arbitrary polygon with optional corner rounding.

    vertices : (N, 2) array of polygon vertices (normalised to CCW internally).
    corner_r : rounding radius applied at convex vertices (0 = sharp corners).

    Returns a list of (fn, arc_length) pairs in the same format as
    _boundary_segments, with fn(t) → (x, y, nx, ny) using inward unit normals.
    """
    pts = np.asarray(vertices, float)
    n   = len(pts)
    # Normalise to CCW
    area2 = sum(pts[i,0]*pts[(i+1)%n,1] - pts[(i+1)%n,0]*pts[i,1] for i in range(n))
    if area2 < 0:
        pts = pts[::-1]

    R = max(0.0, float(corner_r))

    segs = []

    def _lseg(x0, y0, x1, y1, nx, ny):
        L = np.hypot(x1-x0, y1-y0)
        if L < 1e-12:
            return
        def fn(t, x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny):
            return x0+t*(x1-x0), y0+t*(y1-y0), nx, ny
        segs.append((fn, L))

    def _aseg(cx, cy, a0, a1, Rv):          # a1 > a0 (CCW sweep)
        L = (a1 - a0) * Rv
        if L < 1e-12:
            return
        def fn(t, cx=cx, cy=cy, a0=a0, a1=a1, Rv=Rv):
            a = a0 + t*(a1 - a0)
            x, y = cx + Rv*np.cos(a), cy + Rv*np.sin(a)
            return x, y, -np.cos(a), -np.sin(a)   # inward = toward center
        segs.append((fn, L))

    # Per-vertex: compute trim length and arc parameters for convex vertices.
    vtrim  = np.zeros(n)     # trim consumed from each outgoing edge
    varcs  = [None] * n      # (cx, cy, a_start, a_end) or None

    for i in range(n):
        v      = pts[i]
        e_in   = v - pts[(i-1) % n]
        e_out  = pts[(i+1) % n] - v
        l_in   = np.linalg.norm(e_in)
        l_out  = np.linalg.norm(e_out)
        if l_in < 1e-12 or l_out < 1e-12:
            continue
        e_in  /= l_in
        e_out /= l_out

        cross = e_in[0]*e_out[1] - e_in[1]*e_out[0]  # > 0: left turn = convex
        dot   = float(np.dot(e_in, e_out))

        if R > 1e-12 and cross > 1e-6:
            # Half-angle formulas for deflection angle φ = arccos(dot)
            sin_h = np.sqrt(max(0.0, (1.0 - dot) / 2.0))
            cos_h = np.sqrt(max(0.0, (1.0 + dot) / 2.0))
            if cos_h < 1e-9:
                continue     # near-180° turn: skip rounding
            trim = R * sin_h / cos_h    # = R * tan(φ/2)

            arc_s  = v - trim * e_in   # tangent point on incoming edge
            arc_e  = v + trim * e_out  # tangent point on outgoing edge
            # center = arc_s + R * CCW-rot(e_in)
            cen    = arc_s + R * np.array([-e_in[1], e_in[0]])

            a0 = np.arctan2(arc_s[1]-cen[1], arc_s[0]-cen[0])
            a1 = np.arctan2(arc_e[1]-cen[1], arc_e[0]-cen[0])
            if a1 <= a0:
                a1 += 2*np.pi          # ensure CCW sweep

            vtrim[i] = trim
            varcs[i] = (cen[0], cen[1], a0, a1)

    # Validate: trims must not overlap on any edge
    for i in range(n):
        j     = (i + 1) % n
        elen  = np.linalg.norm(pts[j] - pts[i])
        total_trim = vtrim[i] + vtrim[j]
        if total_trim > elen + 1e-6:
            raise ValueError(
                f'corner_r={R:.4g} is too large for edge {i}→{j} '
                f'(edge length {elen:.4g}, total trim {total_trim:.4g}). '
                'Reduce corner_r.')

    # Build segment list: for each edge i→i+1, line segment then arc at i+1
    for i in range(n):
        j      = (i + 1) % n
        e_dir  = pts[j] - pts[i]
        elen   = np.linalg.norm(e_dir)
        if elen < 1e-12:
            continue
        e_dir /= elen

        # Inward normal of this edge (left perp for CCW)
        nx, ny = -e_dir[1], e_dir[0]

        # Line segment: from (pts[i]+vtrim[i]*e_dir) to (pts[j]-vtrim[j]*e_dir)
        ls = pts[i] + vtrim[i] * e_dir
        le = pts[j] - vtrim[j] * e_dir
        _lseg(ls[0], ls[1], le[0], le[1], nx, ny)

        # Arc at vertex j (if rounded)
        if varcs[j] is not None:
            cx, cy, a0, a1 = varcs[j]
            _aseg(cx, cy, a0, a1, R)

    return segs


# ── PolygonRecess ─────────────────────────────────────────────────────────────

class PolygonRecess:
    """Stamped/drawn recess with an arbitrary polygon footprint.

    Uses the same cross-section profile as StampedRecess (top bend → straight
    wall → bottom bend → flat bottom) but the outer footprint is an arbitrary
    polygon instead of a rounded rectangle.

    Parameters
    ----------
    panel       : FlatPanel or PolygonPanel (the host panel)
    polygon     : (N, 2) list/array of (u, v) vertices in panel local coords
                  (T, W frame), defining the CCW footprint boundary
    corner_r    : corner rounding radius (≥ 0) applied at convex vertices
    r_top       : top bend radius (> 0)
    r_bot       : bottom bend radius (> 0)
    depth       : total recess depth  (specify depth OR wall_h)
    wall_h      : straight wall vertical height
    draft_angle : wall tilt from panel-normal direction, degrees (default 0)
    n_boundary  : boundary tessellation resolution
    n_profile   : cross-section tessellation resolution
    """

    def __init__(self, panel, polygon, corner_r=0.0,
                 r_top=3.0, r_bot=3.0, depth=None, wall_h=None,
                 draft_angle=0.0,
                 n_boundary=80, n_profile=20):

        if r_top <= 0 or r_bot <= 0:
            raise ValueError('r_top and r_bot must be positive.')

        draft_angle = float(draft_angle)
        if not (0.0 <= draft_angle < 90.0):
            raise ValueError(f'draft_angle must be in [0, 90) degrees.')
        α         = np.radians(draft_angle)
        bend_drop = (r_top + r_bot) * (1.0 - np.sin(α))

        if depth is not None and wall_h is not None:
            if abs(float(depth) - (bend_drop + float(wall_h))) > 1e-6:
                raise ValueError('Inconsistent depth and wall_h.')
        elif depth is not None:
            wall_h = float(depth) - bend_drop
        elif wall_h is not None:
            depth = bend_drop + float(wall_h)
        else:
            raise ValueError('Specify either depth or wall_h.')

        if wall_h < -1e-9:
            raise ValueError(
                f'wall_h = {wall_h:.4g} < 0. Increase depth or reduce bend radii.')
        wall_h = max(wall_h, 0.0)

        self.panel       = panel
        self.polygon     = np.asarray(polygon, float)
        self.corner_r    = float(corner_r)
        self.r_top       = float(r_top)
        self.r_bot       = float(r_bot)
        self.wall_h      = float(wall_h)
        self.depth       = float(depth)
        self.draft_angle = α
        self.n_boundary  = int(n_boundary)
        self.n_profile   = int(n_profile)

    def faces(self):
        p      = self.panel
        T, W_  = p.T, p.W
        Np     = p.N

        # Centroid of outer polygon → 3-D anchor for flat-bottom fan
        cen_uv = self.polygon.mean(axis=0)
        C3     = p.origin + cen_uv[0]*T + cen_uv[1]*W_

        segs           = _polygon_boundary_segments(self.polygon, self.corner_r)
        bpts, bnorms   = _sample_segs(segs, self.n_boundary)
        us, vs         = _sample_profile(self.r_top, self.wall_h, self.r_bot,
                                         self.draft_angle, self.n_profile)
        NB, NP = self.n_boundary, self.n_profile

        # 3-D sweep grid: boundary point + u * inward_normal_3d + v * panel_normal
        grid = np.empty((NB+1, NP+1, 3))
        for i in range(NB+1):
            bp = p.origin + bpts[i, 0]*T + bpts[i, 1]*W_
            n3 = bnorms[i, 0]*T + bnorms[i, 1]*W_
            for j in range(NP+1):
                grid[i, j] = bp + us[j]*n3 + vs[j]*Np

        out = []
        for i in range(NB):
            for j in range(NP):
                out.append([grid[i, j],   grid[i+1, j],
                            grid[i+1, j+1], grid[i, j+1]])

        # Flat bottom: fan from centroid-at-depth
        centre_bot = C3 + vs[NP]*Np
        for i in range(NB):
            out.append([centre_bot, grid[i, NP], grid[i+1, NP]])

        return out


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Test 1: boundary sampling
    for shape, args in [
        ('circle',   dict(half_a=20, half_b=20, R_c=20, N=80)),
        ('rect',     dict(half_a=30, half_b=20, R_c=5,  N=80)),
        ('square R', dict(half_a=20, half_b=20, R_c=10, N=80)),
    ]:
        pts, norms = _sample_boundary(**args)
        norm_lens = np.linalg.norm(norms, axis=1)
        closed    = np.allclose(pts[0], pts[-1])
        print(f'boundary {shape}: closed={closed}, '
              f'|norm| in [{norm_lens.min():.4f}, {norm_lens.max():.4f}]')
        assert closed, 'boundary not closed'
        assert np.allclose(norm_lens, 1.0, atol=1e-6), 'normals not unit'

    # Test 2: profile sampling (vertical and drafted walls)
    for label, kwargs in [
        ('r3/w5/r3 α=0°',  dict(r_top=3, wall_h=5, r_bot=3, draft_angle=0.0, N=20)),
        ('r5/w0/r5 α=0°',  dict(r_top=5, wall_h=0, r_bot=5, draft_angle=0.0, N=20)),
        ('r3/w5/r3 α=15°', dict(r_top=3, wall_h=5, r_bot=3, draft_angle=np.radians(15), N=20)),
        ('r4/w3/r4 α=30°', dict(r_top=4, wall_h=3, r_bot=4, draft_angle=np.radians(30), N=20)),
    ]:
        us, vs = _sample_profile(**kwargs)
        r_top, r_bot, wall_h, α = (kwargs['r_top'], kwargs['r_bot'],
                                    kwargs['wall_h'], kwargs['draft_angle'])
        sin_a, cos_a = np.sin(α), np.cos(α)
        exp_u_end = (r_top + r_bot)*cos_a + wall_h*np.tan(α)
        exp_v_end = -((r_top + r_bot)*(1 - sin_a) + wall_h)
        print(f'profile {label}: u[0]={us[0]:.4f} v[0]={vs[0]:.4f} '
              f'u[-1]={us[-1]:.4f} (exp {exp_u_end:.4f}) '
              f'v[-1]={vs[-1]:.4f} (exp {exp_v_end:.4f})')
        assert abs(us[0]) < 1e-9 and abs(vs[0]) < 1e-9, 'profile start wrong'
        assert abs(us[-1] - exp_u_end) < 1e-5, f'profile u end wrong: {us[-1]} != {exp_u_end}'
        assert abs(vs[-1] - exp_v_end) < 1e-5, f'profile v end wrong: {vs[-1]} != {exp_v_end}'

    # Test 3: full recess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sys.path.insert(0, '/home/yang/Bracket')
    from sheet_metal.primitives import FlatPanel

    panel = FlatPanel([0,0,0], [1,0,0], [0,1,0], 120, 80, t=2.0)

    for label, kwargs in [
        ('rect_recess',     dict(cx=60, cy=40, width=60, height=40, corner_r=8,
                                 r_top=3, r_bot=3, wall_h=5)),
        ('circ_recess',     dict(cx=60, cy=40, width=40, height=40, corner_r=20,
                                 r_top=3, r_bot=3, wall_h=4)),
        ('drafted_recess',  dict(cx=60, cy=40, width=60, height=40, corner_r=12,
                                 r_top=3, r_bot=3, wall_h=5, draft_angle=20.0)),
    ]:
        rec = StampedRecess(panel, **kwargs)
        fs  = rec.faces()
        n_sweep = rec.n_boundary * rec.n_profile
        n_bot   = rec.n_boundary
        print(f'StampedRecess {label}: {len(fs)} faces '
              f'(expected sweep={n_sweep} + bottom={n_bot} = {n_sweep+n_bot})')
        assert len(fs) == n_sweep + n_bot

    # Test 4: polygon boundary segments
    import math
    hex_verts = np.array([[50+30*math.cos(i*math.pi/3), 40+30*math.sin(i*math.pi/3)]
                           for i in range(6)])
    for label, cr in [('hex sharp', 0.0), ('hex rounded r4', 4.0)]:
        segs = _polygon_boundary_segments(hex_verts, cr)
        pts2, nrms = _sample_segs(segs, 120)
        closed = np.allclose(pts2[0], pts2[-1])
        nlen   = np.linalg.norm(nrms, axis=1)
        print(f'polygon boundary {label}: closed={closed}, '
              f'|norm| in [{nlen.min():.4f}, {nlen.max():.4f}]')
        assert closed, 'polygon boundary not closed'
        assert np.allclose(nlen, 1.0, atol=1e-5), 'polygon normals not unit'

    # Test 5: PolygonRecess
    tri_verts = [[60, 20], [100, 65], [20, 65]]   # triangle on panel
    for label, kwargs in [
        ('tri_sharp',   dict(corner_r=0.0, r_top=3, r_bot=3, wall_h=5)),
        ('tri_rounded', dict(corner_r=6.0, r_top=3, r_bot=3, wall_h=5)),
        ('hex_drafted', dict(corner_r=3.0, r_top=3, r_bot=3, wall_h=4, draft_angle=15.0)),
    ]:
        verts = hex_verts if 'hex' in label else tri_verts
        rec = PolygonRecess(panel, verts, **kwargs)
        fs  = rec.faces()
        exp = rec.n_boundary * rec.n_profile + rec.n_boundary
        print(f'PolygonRecess {label}: {len(fs)} faces (expected {exp})')
        assert len(fs) == exp, f'{len(fs)} != {exp}'

    print('All tests passed.')
