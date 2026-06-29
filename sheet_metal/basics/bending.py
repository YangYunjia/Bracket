import numpy as np
from .maths import n


# ── BendArc ───────────────────────────────────────────────────────────────────

class BendArc:
    """Full-edge cylindrical arc (mid-surface strip)."""

    def __init__(self, p0, p1, T_out, N_bend, alpha, r_n):
        self.p0     = np.asarray(p0, float)
        self.p1     = np.asarray(p1, float)
        self.T_out  = n(T_out)
        self.N_bend = n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)


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
                 blend_r=None, taper='arc'):
        self.p0     = np.asarray(p0_full, float)
        self.p1     = np.asarray(p1_full, float)
        self.L      = float(np.linalg.norm(self.p1 - self.p0))
        self.s0     = float(s_start)
        self.s1     = float(s_end)
        self.T_out  = n(T_out)
        self.N_bend = n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        self.blend  = float(blend_r) if blend_r is not None else float(r_n)
        self.taper  = taper

