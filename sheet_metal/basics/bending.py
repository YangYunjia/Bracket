import numpy as np
from .maths import n


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

    def __init__(self, name, p0, p1,
                 T_in, W_in, N_bend,
                 alpha, r_n,
                 blend_r, blend_mode='arc'):
        self.name   = name
        self.p0     = np.asarray(p0, float)
        self.p1     = np.asarray(p1, float)

        self.L      = float(np.linalg.norm(self.p1 - self.p0))

        self.T_in  = n(T_in)
        self.T_out = n(np.sin(alpha)*N_bend + np.cos(alpha)*T_in)
        self.W     = n(W_in)
        self.N_bend = n(N_bend)
        self.N_out  = n(np.cos(alpha)*N_bend - np.sin(alpha)*T_in)

        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        if isinstance(blend_r, tuple):
            self.blend_r = (float(blend_r[0]), float(blend_r[1]))
        else:
            self.blend_r = (float(blend_r), float(blend_r))
        self.blend_mode  = blend_mode

        self.eff_r_n = max(self.r_n, max(self.blend_r) / self.alpha) if self.alpha > 1e-9 else self.r_n # ensure the length of the arc is at least the radius of the blend
        
        def shift_point(p):
            return p + self.eff_r_n * ((1 - np.cos(alpha)) * N_bend + np.sin(alpha) * T_in) if alpha > 1e-9 else p + self.r_n * T_in
        
        self.p0_out = shift_point(self.p0)
        self.p1_out = shift_point(self.p1)

# ── BendArc ───────────────────────────────────────────────────────────────────

class BendArc(PartialBendArc):
    """Full-edge cylindrical arc (mid-surface strip).
    
    p0 : parent edge start point
    p1 : parent edge end point
    T_in : parent edge outward tangent
    N_bend : parent panel normal
    alpha : arc angle
    r_n : bending radius
    
    """

    def __init__(self, name, p0, p1, T_in, W_in, N_bend, alpha, r_n):
        super().__init__(name, p0, p1, T_in, W_in, N_bend, alpha, r_n, blend_r=0.0, blend_mode='linear')