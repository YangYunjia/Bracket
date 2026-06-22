"""
sheet_metal.recess — stamped/drawn recess feature.

Geometry (cross-section, outside → inside):
  panel surface → top bend (r_top) → straight wall (wall_h)
                → bottom bend (r_bot) → flat bottom

depth = r_top + wall_h + r_bot
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


def _sample_boundary(half_a, half_b, R_c, N):
    """Sample N+1 points on the rounded-rect boundary (closed: pts[N]==pts[0]).

    Returns pts (N+1,2) and norms (N+1,2) — inward unit normals.
    """
    segs   = _boundary_segments(half_a, half_b, R_c)
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


# ── cross-section profile ────────────────────────────────────────────────────

def _sample_profile(r_top, wall_h, r_bot, N):
    """Sample N+1 (u, v) pairs along the cross-section profile.

    u : inward offset from outer footprint boundary (>= 0)
    v : depth below panel surface (<= 0)

    Sequence: top-bend arc → straight wall → bottom-bend arc.
    """
    L_top  = 0.5 * np.pi * r_top
    L_wall = wall_h
    L_bot  = 0.5 * np.pi * r_bot
    total  = L_top + L_wall + L_bot

    us = np.empty(N+1)
    vs = np.empty(N+1)

    for i in range(N+1):
        s = i * total / N
        if s <= L_top + 1e-12:
            θ = (s / L_top) * (0.5*np.pi) if L_top > 1e-12 else 0.5*np.pi
            θ = min(θ, 0.5*np.pi)
            us[i] =  r_top * np.sin(θ)
            vs[i] = -r_top * (1.0 - np.cos(θ))
        elif s <= L_top + L_wall + 1e-12:
            tw    = s - L_top
            us[i] = r_top
            vs[i] = -(r_top + tw)
        else:
            sb = s - L_top - L_wall
            θ  = (sb / L_bot) * (0.5*np.pi) if L_bot > 1e-12 else 0.5*np.pi
            θ  = min(θ, 0.5*np.pi)
            us[i] = r_top + r_bot * (1.0 - np.cos(θ))
            vs[i] = -(r_top + wall_h + r_bot * np.sin(θ))

    return us, vs


# ── StampedRecess ─────────────────────────────────────────────────────────────

class StampedRecess:
    """Stamped / drawn recess on a flat panel.

    The recess footprint is a rounded rectangle in the panel's (T, W) plane.
    width == height and corner_r == width/2 gives a circular recess.

    Parameters
    ----------
    panel      : FlatPanel
    cx, cy     : center of recess in panel local coords (from panel.origin,
                 along panel.T and panel.W respectively)
    width      : footprint outer width  (along panel.T)
    height     : footprint outer height (along panel.W)
    corner_r   : corner radius; clamped to min(width,height)/2
    r_top      : top bend radius  (must be > 0)
    r_bot      : bottom bend radius (must be > 0)
    depth      : total depth = r_top + wall_h + r_bot  (specify depth OR wall_h)
    wall_h     : straight wall height (derived from depth if omitted)
    n_boundary : tessellation resolution around boundary
    n_profile  : tessellation resolution along cross-section
    """

    def __init__(self, panel, cx, cy, width, height, corner_r,
                 r_top, r_bot, depth=None, wall_h=None,
                 n_boundary=80, n_profile=20):

        # ── resolve depth / wall_h ────────────────────────────────────────
        if depth is not None and wall_h is not None:
            if abs(depth - (r_top + wall_h + r_bot)) > 1e-6:
                raise ValueError(
                    f'Inconsistent: depth={depth} != r_top+wall_h+r_bot='
                    f'{r_top+wall_h+r_bot}')
        elif depth is not None:
            wall_h = depth - r_top - r_bot
        elif wall_h is not None:
            depth = r_top + wall_h + r_bot
        else:
            raise ValueError('Specify either depth or wall_h.')

        # ── clamp corner radius ───────────────────────────────────────────
        corner_r = min(float(corner_r), min(width, height) / 2.0)

        # ── validate ──────────────────────────────────────────────────────
        if r_top <= 0 or r_bot <= 0:
            raise ValueError('r_top and r_bot must be positive.')
        if wall_h < -1e-9:
            raise ValueError(
                f'wall_h = depth - r_top - r_bot = {wall_h:.4g} < 0. '
                'Increase depth or reduce bend radii.')
        wall_h = max(wall_h, 0.0)
        inset = r_top + r_bot
        if corner_r < inset - 1e-9:
            raise ValueError(
                f'corner_r ({corner_r:.4g}) < r_top+r_bot ({inset:.4g}): '
                'profile would self-intersect at corners. '
                'Increase corner_r or reduce bend radii.')
        if width <= 2*inset or height <= 2*inset:
            raise ValueError(
                f'Footprint ({width}×{height}) too small: '
                f'need both dims > 2*(r_top+r_bot) = {2*inset:.4g}.')
        # panel bounds check
        if (cx - width/2  < -1e-6 or cx + width/2  > panel.length + 1e-6 or
            cy - height/2 < -1e-6 or cy + height/2 > panel.width  + 1e-6):
            raise ValueError(
                f'Recess footprint [{cx-width/2:.3g},{cx+width/2:.3g}] × '
                f'[{cy-height/2:.3g},{cy+height/2:.3g}] lies outside '
                f'panel ({panel.length}×{panel.width}).')

        self.panel      = panel
        self.cx         = float(cx)
        self.cy         = float(cy)
        self.half_a     = width  / 2.0
        self.half_b     = height / 2.0
        self.corner_r   = float(corner_r)
        self.r_top      = float(r_top)
        self.r_bot      = float(r_bot)
        self.wall_h     = float(wall_h)
        self.depth      = float(depth)
        self.n_boundary = int(n_boundary)
        self.n_profile  = int(n_profile)

    def faces(self):
        p     = self.panel
        T, W_ = p.T, p.W
        Np    = p.N                                  # panel outward normal
        C3    = p.origin + self.cx*T + self.cy*W_   # recess centre in 3D

        bpts, bnorms = _sample_boundary(
            self.half_a, self.half_b, self.corner_r, self.n_boundary)
        us, vs = _sample_profile(
            self.r_top, self.wall_h, self.r_bot, self.n_profile)

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

    # Test 2: profile sampling
    for label, kwargs in [
        ('r3/w5/r3',  dict(r_top=3, wall_h=5, r_bot=3, N=20)),
        ('r5/w0/r5',  dict(r_top=5, wall_h=0, r_bot=5, N=20)),
    ]:
        us, vs = _sample_profile(**kwargs)
        print(f'profile {label}: u[0]={us[0]:.4f} v[0]={vs[0]:.4f} '
              f'u[-1]={us[-1]:.4f} v[-1]={vs[-1]:.4f}')
        assert abs(us[0]) < 1e-9 and abs(vs[0]) < 1e-9, 'profile start wrong'
        r_top, r_bot = kwargs['r_top'], kwargs['r_bot']
        assert abs(us[-1] - (r_top+r_bot)) < 1e-6, 'profile u end wrong'
        assert abs(vs[-1] - (-(r_top+kwargs['wall_h']+r_bot))) < 1e-6, 'profile v end wrong'

    # Test 3: full recess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sys.path.insert(0, '/home/yang/Bracket')
    from sheet_metal.primitives import FlatPanel

    panel = FlatPanel([0,0,0], [1,0,0], [0,1,0], 120, 80, t=2.0)

    for label, kwargs in [
        ('rect_recess',   dict(cx=60, cy=40, width=60, height=40, corner_r=8,
                               r_top=3, r_bot=3, wall_h=5)),
        ('circ_recess',   dict(cx=60, cy=40, width=40, height=40, corner_r=20,
                               r_top=3, r_bot=3, wall_h=4)),
    ]:
        rec = StampedRecess(panel, **kwargs)
        fs  = rec.faces()
        n_sweep = rec.n_boundary * rec.n_profile
        n_bot   = rec.n_boundary
        print(f'StampedRecess {label}: {len(fs)} faces '
              f'(expected sweep={n_sweep} + bottom={n_bot} = {n_sweep+n_bot})')
        assert len(fs) == n_sweep + n_bot

    print('All tests passed.')
