"""
sheet_metal.primitives — geometry primitives for mid-surface shell models.

FlatPanel  : flat rectangular panel
BendArc    : full-edge cylindrical arc strip
PartialBendArc : partial-edge arc with smooth width taper
"""

import numpy as np


def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v


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
    """

    def __init__(self, origin, T, W, length, width, t=2.0, name='P'):
        self.origin = np.asarray(origin, float)
        self.T      = _n(T)
        self.W      = _n(W)
        self.length = float(length)
        self.width  = float(width)
        self.t      = float(t)
        self.name   = str(name)

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
