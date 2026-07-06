'''
Basic 3-D curves.

All 3-D curves works in the 3-D world coordinate system.
'''


import numpy as np
from typing import Sequence
from .maths import n
from scipy.interpolate import BSpline as _BSpline


# ── Curve3D ───────────────────────────────────────────────────────────────────

class Curve3D:
    """Lightweight 3-D world-space curve.

    Constructors:
        Curve3D.line(p0, p1)
        Curve3D.polyline(pts)
        Curve3D.arc(center, axis, radius, p_start, p_end)
        Curve3D.bspline(control_pts, degree=3)

    All expose:
        sample(n)         → (n, 3) float array, arc-length-uniform
        tangent_at(t)     → unit tangent at parameter t ∈ [0, 1]
        point_at_frac(f)  → 3-D point at arc-length fraction f ∈ [0, 1]
        tangent_at_frac(f)→ unit tangent at arc-length fraction f
        insert_fracs(fs)  → new polyline with split points at arc-length fractions
        reversed()        → same curve with arc-length parameter flipped (t ↦ 1−t)
        endpoints()       → (p0, p1) 3-D arrays (t=0, t=1 points)
        arc_length()      → float


    """
    def __init__(self):
        pass

    @staticmethod
    def line(p0: Sequence[float], p1: Sequence[float]) -> 'Polyline3D':
        return Polyline3D([p0, p1])

    @staticmethod
    def polyline(pts: Sequence[Sequence[float]]) -> 'Polyline3D':
        return Polyline3D(pts)

    @staticmethod
    def arc(center: Sequence[float], axis: Sequence[float], radius: float, p_start: Sequence[float], p_end: Sequence[float]) -> 'Arc3D':
        return Arc3D(center, axis, radius, p_start, p_end)

    @staticmethod
    def bspline(control_pts: Sequence[Sequence[float]], degree: int = 3) -> 'BSpline3D':
        return BSpline3D(control_pts, degree)


# ── BaseCurve3D ────────────────────────────────────────────────────────────────
class BaseCurve3D:

    def __init__(self, pts: Sequence[Sequence[float]]):
        self._pts = [np.asarray(p, float) for p in pts]

    def reversed(self):
        return self.__class__(list(reversed(self._pts)))

    def shift(self, vec: Sequence[float]) -> 'BaseCurve3D':
        return self.__class__([p + vec for p in self._pts])

    def segment_lengths(self):
        raise NotImplementedError('Segment lengths of base curve are not implemented.')

    def endpoints(self):
        raise NotImplementedError('Endpoints of base curve are not implemented.')

    def length(self):
        raise NotImplementedError('Length of base curve is not implemented.')

    def interpolate_raw(self, t: float) -> np.ndarray:
        raise NotImplementedError('Interpolation of base curve is not implemented.')

    def tangent_at(self, t: float) -> np.ndarray:
        raise NotImplementedError('Tangent at of base curve is not implemented.')

    def frac2raw(self, f: float) -> float:
        raise NotImplementedError('Fraction to raw parameter of base curve is not implemented.')

    def interpolate(self, f: float) -> np.ndarray:
        return self.interpolate_raw(self.frac2raw(f))

    def tangent_at_frac(self, f: float) -> np.ndarray:
        return self.tangent_at(self.frac2raw(f))

    def insert_fracs(self, fracs: Sequence[float], tol: float = 1e-9) -> 'BaseCurve3D':
        raise NotImplementedError('Insert fractions of base curve is not implemented.')

    def sample(self, n: int, uniform: bool = True):
        raise NotImplementedError('Sample is only used for numpy backend, and is deprecated.')
        

class Polyline3D(BaseCurve3D):
    def __init__(self, pts: Sequence[Sequence[float]]):
        pts = [np.asarray(p, float) for p in pts]
        # remove duplicate points
        out = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - out[-1]) > 1e-12:
                out.append(p)
        if len(out) < 2: raise ValueError('polyline needs at least 2 distinct points.')
        super().__init__(out)

    def insert_fracs(self, fracs: Sequence[float], tol: float = 1e-9) -> 'Polyline3D':
        """Return a polyline with extra vertices at arc-length fractions."""
        fracs = sorted({float(f) for f in fracs
                        if tol < float(f) < 1.0 - tol})
        pts = []
        for f in [0.0] + fracs + [1.0]:
            p = np.asarray(self.interpolate(f), float)
            if not pts or np.linalg.norm(p - pts[-1]) > tol:
                pts.append(p)
        return Polyline3D(pts)

    def sharp_points(self):
        return [np.asarray(p, float).copy() for p in self._pts]

    def segment_lengths(self):
        '''
        Segment lengths of the polyline.
        
        Returns
        -------
        pts : list of 3-D points
        segs : list of segment lengths
        '''

        pts = self.sharp_points()
        segs = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        return pts, segs   # (N, 3) and (N-1,) arrays

    def length(self):
        return sum(self.segment_lengths()[1])

    def interpolate_raw(self, t):
        """
        Interpolate curve at parameter t (scalar or array).
        
        Parameters
        ----------
        t : scalar or array-like
            Parameter value(s) in [0, 1]
            
        Returns
        -------
        out : array-like
            Evaluated curve points (shape: (..., 3))
        """
        t = np.asarray(t, float)
        pts, segs = self.segment_lengths()
        n_seg = len(segs)
        t_flat = np.asarray(t, float).ravel()
        out = np.empty((t_flat.size, 3))
        for k, tv in enumerate(t_flat):
            seg_f = float(np.clip(tv, 0.0, 1.0)) * n_seg
            i = min(int(np.floor(seg_f)), n_seg - 1)
            if seg_f >= n_seg - 1e-12:
                i = n_seg - 1
                u = 1.0
            else:
                u = seg_f - i
            out[k] = pts[i] + u * (pts[i + 1] - pts[i])
        return out.reshape((*t.shape, 3)) if t.ndim else out[0]

    def frac2raw(self, f):
        """
        Map arc-length fraction f ∈ [0,1] to raw parameter t ∈ [0,1].
        
        Parameters
        ----------
        f : scalar or array-like
            Arc-length fraction value(s) in [0, 1]
            
        Returns
        -------
        t : scalar or array-like
            Parameter value(s) in [0, 1]
        """
        pts, segs = self.segment_lengths()
        total = sum(segs)
        if total < 1e-15:
            return 0.0
        s = f * total
        acc = 0.0
        for i, sl in enumerate(segs):
            if sl < 1e-15:
                continue
            if s <= acc + sl + 1e-12:
                return (i + (s - acc) / sl) / (len(pts) - 1)
            acc += sl
        return 1.0

    def tangent_at(self, t):
        '''
        Unit tangent at parameter t (scalar or array).
        
        Parameters
        ----------
        t : scalar or array-like
            Parameter value(s) in [0, 1]
            
        Returns
        -------
        out : array-like
            Unit tangent at parameter t (shape: (..., 3))
        '''
        pts, segs = self.segment_lengths()
        n_seg = len(segs)
        seg_f = float(np.clip(t, 0.0, 1.0)) * n_seg
        i = min(int(np.floor(seg_f)), n_seg - 1)
        if seg_f >= n_seg - 1e-12:
            i = max(n_seg - 1, 0)
        d = pts[i + 1] - pts[i]
        if np.linalg.norm(d) < 1e-15 and i > 0:
            d = pts[i] - pts[i - 1]
        return n(d)

    def endpoints(self):
        return self._pts[0].copy(), self._pts[-1].copy()


class Arc3D(BaseCurve3D):
    def __init__(self, center: Sequence[float], axis: Sequence[float], radius: float, p_start: Sequence[float], p_end: Sequence[float]):
        """Circular arc.

        Parameters
        ----------
        center  : 3-D centre of the circle
        axis    : normal to the circle plane (CCW positive)
        radius  : circle radius
        p_start : 3-D point on the arc (t=0); must lie on the circle
        p_end   : 3-D point on the arc (t=1); must lie on the circle
        """
        self.center = np.asarray(center, float)
        self.axis = n(np.asarray(axis, float))
        self.radius = float(radius)
        self.p_start = np.asarray(p_start, float)
        self.p_end = np.asarray(p_end, float)

    def reversed(self) -> 'Arc3D':
        return Arc3D(self.center, -self.axis, self.radius, self.p_end, self.p_start)

    def shift(self, vec: Sequence[float]) -> 'Arc3D':
        return Arc3D(self.center + vec, self.axis, self.radius, self.p_start + vec, self.p_end + vec)

    def segment_lengths(self):
        raise NotImplementedError('Segment lengths of arc are not implemented.')

    def interpolate_raw(self, t):
        """
        Interpolate curve at parameter t (scalar or array).
        
        Parameters
        ----------
        t : scalar or array-like
            Parameter value(s) in [0, 1]
        """
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

    def frac2raw(self, f):
        return f

    def tangent_at(self, t):
        eps = 1e-5
        t1  = min(t + eps, 1.0)
        t0  = max(t - eps, 0.0)
        return n(self.interpolate_raw(np.array([t1]))[0]
                    - self._arc_eval(np.array([t0]))[0])

    def endpoints(self):
        return self.p_start.copy(), self.p_end.copy()

class BSpline3D(BaseCurve3D):

    def __init__(self, control_pts: Sequence[Sequence[float]], degree: int = 3):
        """3-D B-spline curve through (or near) control_pts.

        Parameters
        ----------
        control_pts : (M, 3) control polygon in world coordinates
        degree      : spline degree (default 3 = cubic)
        """

        self.control_pts = np.asarray(control_pts, float)
        self.degree = int(degree)
        if self.control_pts.ndim != 2 or self.control_pts.shape[1] != 3:
            raise ValueError('control_pts must be (M, 3).')
        if len(self.control_pts) < self.degree + 1:
            raise ValueError(f'Need at least {self.degree+1} control points for degree {self.degree}.')
        self.n_int = len(self.control_pts) - self.degree - 1
        self.intr = (np.linspace(0.0, 1.0, self.n_int + 2)[1:-1]
                     if self.n_int > 0 else np.empty(0))
        self.knots = np.concatenate([np.zeros(self.degree+1), self.intr, np.ones(self.degree+1)])
        self.sps = [_BSpline(self.knots, self.control_pts[:, d], self.degree) for d in range(3)] # three directions x, y, z
        self.dsps = [sp.derivative() for sp in self.sps]

    def reversed(self) -> 'BSpline3D':
        return BSpline3D(self.control_pts[::-1], self.degree)

    def interpolate_raw(self, t):
        """
        Interpolate curve at parameter t (scalar or array).
        
        Parameters
        ----------
        t : scalar or array-like
            Parameter value(s) in [0, 1]
        """
        return np.stack([sp(t) for sp in self.sps], axis=-1)

    def tangent_at(self, t):
        """Unit tangent at parameter t ∈ [0, 1] (scalar)."""
        d = np.array([dsp(t) for dsp in self._dsps])
        return n(d)

    def endpoints(self):
        """Return (p0, p1) — the t=0 and t=1 world-space points."""
        pts = self.interpolate_raw(np.array([0.0, 1.0]))
        return pts[0].copy(), pts[1].copy()
