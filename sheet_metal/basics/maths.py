import numpy as np

def n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v

def rot(v, k, deg):
    v, k = np.asarray(v, float), n(np.asarray(k, float))
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1.0 - np.cos(t))

# ── polygon helpers ───────────────────────────────────────────────────────────

def signed_area2(pts):
    """Twice the signed area (positive = CCW)."""
    n = len(pts)
    return float(sum(pts[i, 0]*pts[(i+1)%n, 1] - pts[(i+1)%n, 0]*pts[i, 1]
                     for i in range(n)))


def pt_in_tri(p, a, b, c):
    """True if point p is strictly inside triangle (a, b, c)."""
    def _s(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = _s(p, a, b), _s(p, b, c), _s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))