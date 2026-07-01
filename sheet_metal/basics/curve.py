import numpy as np
from .maths import n


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
