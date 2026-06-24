"""
generate_loft_samples.py — twisted L- and U-shape bracket examples using LoftPanel.

Run inside the occ312 conda environment:
    conda run -n occ312 python generate_loft_samples.py

Outputs:
  output_loft/<name>.png       — 3-view render (ISO / Side / Top)
  output_loft/step/<name>.step — STEP AP203 B-Rep file
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sheet_metal_occ import OCC_BracketModel
from sheet_metal import Curve3D


OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_loft')
STEP_OUT = os.path.join(OUT, 'step')
os.makedirs(OUT,      exist_ok=True)
os.makedirs(STEP_OUT, exist_ok=True)

VIEWS = [(28, -50, 'ISO'), (0, 90, 'Side'), (90, -90, 'Top')]


# ─── render / save helper ─────────────────────────────────────────────────────

def save(m, name, title):
    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim, label) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        m.render(ax, elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
    fig.suptitle(f'[LoftPanel] {title}', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    step_path = os.path.join(STEP_OUT, f'{name}.step')
    ok  = m.export_step(step_path)
    tag = 'step✓' if ok else 'step!'
    print(f'  {name}.png  [{tag}]')


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _rot(v, k, deg):
    """Rodrigues rotation of 3-D vector v around unit axis k by deg degrees."""
    v, k = np.asarray(v, float), np.asarray(k, float)
    t = np.radians(deg)
    return (v * np.cos(t)
            + np.cross(k, v) * np.sin(t)
            + k * np.dot(k, v) * (1.0 - np.cos(t)))


def _free_line(root_p0, root_p1, T_child, arm_length, twist_deg,
               width_scale=1.0):
    """Straight free edge for a twisted / tapered LoftPanel arm.

    Parameters
    ----------
    root_p0, root_p1 : 3-D endpoints of the root edge
    T_child          : unit arm-extension direction
    arm_length       : mm along T_child from root to free edge
    twist_deg        : rotation of the width axis around T_child (degrees)
    width_scale      : free-edge width = root_width * width_scale  (1 = same)
    """
    root_p0 = np.asarray(root_p0, float)
    root_p1 = np.asarray(root_p1, float)
    T_child = np.asarray(T_child, float) / np.linalg.norm(T_child)

    mid_root  = (root_p0 + root_p1) / 2.0
    W_root    = root_p1 - root_p0
    half_w    = np.linalg.norm(W_root) / 2.0
    W_unit    = W_root / (2.0 * half_w)

    mid_free  = mid_root + arm_length * T_child
    W_end     = _rot(W_unit, T_child, twist_deg)
    W_end     = W_end / np.linalg.norm(W_end)
    hw_free   = half_w * width_scale

    return Curve3D.line(mid_free - hw_free * W_end,
                        mid_free + hw_free * W_end)


def _free_arc(root_p0, root_p1, T_child, arm_length, bulge_sign=+1):
    """Semicircular free edge centred on the straight arm axis.

    bulge_sign=+1 → arc bulges toward +T×W (outward face of the arm).
    """
    root_p0   = np.asarray(root_p0, float)
    root_p1   = np.asarray(root_p1, float)
    T_child   = np.asarray(T_child, float) / np.linalg.norm(T_child)

    mid_root  = (root_p0 + root_p1) / 2.0
    W_root    = root_p1 - root_p0
    half_w    = np.linalg.norm(W_root) / 2.0
    W_unit    = W_root / (2.0 * half_w)

    mid_free  = mid_root + arm_length * T_child
    p_start   = mid_free - half_w * W_unit     # arc start = left corner
    p_end     = mid_free + half_w * W_unit     # arc end   = right corner
    center    = mid_free

    # Axis direction: choose so that the arc midpoint bulges toward
    # bulge_sign * (T_child × W_unit)
    outward   = np.cross(T_child, W_unit)
    # With axis = -bulge_sign * T_child, the 90° midpoint lies along +outward
    # when bulge_sign=+1 (verified by Rodrigues).
    arc_axis  = -float(bulge_sign) * T_child

    return Curve3D.arc(center=center, axis=arc_axis, radius=half_w,
                       p_start=p_start, p_end=p_end)


# ─── sample builders ──────────────────────────────────────────────────────────
# All use base with T=[1,0,0], W=[0,1,0], so N=[0,0,1].
# Standard inner_r=4, t=2  →  r_n=5.
# A 90° bend from edge 3 (top, v=W) produces:
#   T_child=[0,0,1], W_child=[1,0,0],
#   root edge from [0, W+r_n, r_n] to [L, W+r_n, r_n]
# A 90° bend from edges 0/1 (left/right, u=0/L) produces:
#   T_child=[0,0,1], W_child=[0,1,0],
#   root edge from [±r_n, 0, r_n] to [±r_n, W, r_n]


# ── L-shape examples ──────────────────────────────────────────────────────────

def twisted_l_90(base_l=80, base_w=50, arm_h=60, t=2.0, inner_r=4.0):
    """L-bracket: arm twists exactly 90° from root (horizontal) to free end (vertical).

    Root edge runs along X (80 mm wide); free edge runs along Y (same 80 mm
    width), producing a ruled ribbon that rotates 90° over its 60 mm height.
    """
    r_n  = inner_r + t / 2          # = 5
    root = ([0,          base_w + r_n, r_n],
            [base_l,     base_w + r_n, r_n])   # [0,55,5]→[80,55,5]
    T_ch = [0, 0, 1]

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend_loft(base, edge_idx=3, angle_deg=90,
                  free_curve=_free_line(*root, T_ch, arm_h, twist_deg=90))
    return m


def twisted_l_tapered(base_l=80, base_w=50, arm_h=60, t=2.0, inner_r=4.0):
    """L-bracket: arm twists 45° and tapers to 60 % of the root width.

    Combines a diagonal twist with narrowing, so the free edge is a short
    oblique line while the root is the full 80 mm width.
    """
    r_n  = inner_r + t / 2
    root = ([0,      base_w + r_n, r_n],
            [base_l, base_w + r_n, r_n])
    T_ch = [0, 0, 1]

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend_loft(base, edge_idx=3, angle_deg=90,
                  free_curve=_free_line(*root, T_ch, arm_h,
                                        twist_deg=45, width_scale=0.6))
    return m


def twisted_l_arc_free(base_l=100, base_w=60, arm_h=60, t=2.0, inner_r=4.0):
    """L-bracket: arm free edge is a semicircular arc (fan-blade shape).

    Root is a straight 100 mm line; free end is a 100 mm diameter semicircle
    bulging outward from the arm face.  No twist — only the edge shape changes.
    """
    r_n  = inner_r + t / 2
    root = ([0,       base_w + r_n, r_n],
            [base_l,  base_w + r_n, r_n])
    T_ch = [0, 0, 1]

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend_loft(base, edge_idx=3, angle_deg=90,
                  free_curve=_free_arc(*root, T_ch, arm_h, bulge_sign=+1))
    return m


def twisted_l_bspline_free(base_l=80, base_w=50, arm_h=70, t=2.0, inner_r=4.0):
    """L-bracket: arm has a sinuous B-spline free edge.

    The free edge is prescribed by 4 control points forming an S-curve across
    the arm width, giving the arm a scalloped profile at its free end.
    """
    r_n   = inner_r + t / 2
    root  = ([0,      base_w + r_n, r_n],
             [base_l, base_w + r_n, r_n])
    T_ch  = np.array([0, 0, 1], float)

    root_p0 = np.array(root[0], float)
    root_p1 = np.array(root[1], float)
    mid_root = (root_p0 + root_p1) / 2.0
    W_unit   = (root_p1 - root_p0) / np.linalg.norm(root_p1 - root_p0)
    half_w   = np.linalg.norm(root_p1 - root_p0) / 2.0

    # S-curve control points in the free-edge plane (Z = arm_h above root mid)
    # Left-to-right: dip then rise to form a gentle S
    offs = np.array([-1.0, -0.33, 0.33, 1.0]) * half_w   # along W_unit
    ups  = np.array([ 0.0,  0.6, -0.6,  0.0]) * half_w   # along T_ch (Z variation)
    cp3d = np.array([
        (mid_root + root_p0) / 2 + arm_h * T_ch,     # left endpoint
        mid_root + offs[1] * W_unit + ups[1] * T_ch + arm_h * T_ch,
        mid_root + offs[2] * W_unit + ups[2] * T_ch + arm_h * T_ch,
        (mid_root + root_p1) / 2 + arm_h * T_ch,     # right endpoint
    ], float)
    # Adjust endpoints to lie exactly at the left/right extremes
    cp3d[0]  = root_p0 + arm_h * T_ch
    cp3d[-1] = root_p1 + arm_h * T_ch

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend_loft(base, edge_idx=3, angle_deg=90,
                  free_curve=Curve3D.bspline(cp3d, degree=3))
    return m


# ── U-shape examples ──────────────────────────────────────────────────────────
# Arms extend from edges 0 (u=0, left) and 1 (u=L, right).
# root_left:  [-r_n, 0, r_n] → [-r_n, W, r_n]
# root_right: [L+r_n, 0, r_n] → [L+r_n, W, r_n]
# T_child = [0, 0, 1] for both; W_child = [0, 1, 0].


def twisted_u_outward(base_l=80, base_w=50, arm_h=50, t=2.0, inner_r=4.0):
    """U-bracket: both arms twist 90° outward so their free edges are horizontal.

    Left arm free end points in -X; right arm free end points in +X.
    The U "opens up" at the top into a wide T-shape when viewed from the side.
    """
    r_n   = inner_r + t / 2
    T_ch  = [0, 0, 1]
    root_L = ([-r_n, 0,        r_n], [-r_n, base_w, r_n])
    root_R = ([base_l + r_n, 0, r_n], [base_l + r_n, base_w, r_n])

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    # Left arm: root W=[0,1,0] → twisted +90° → W_end=[-1,0,0] (outward)
    m.extend_loft(base, edge_idx=0, angle_deg=90,
                  free_curve=_free_line(*root_L, T_ch, arm_h, twist_deg=+90))
    # Right arm: root W=[0,1,0] → twisted -90° → W_end=[+1,0,0] (outward)
    m.extend_loft(base, edge_idx=1, angle_deg=90,
                  free_curve=_free_line(*root_R, T_ch, arm_h, twist_deg=-90))
    return m


def twisted_u_one_twisted(base_l=80, base_w=50, arm_h=50, t=2.0, inner_r=4.0):
    """U-bracket: left arm is a standard flat panel; right arm twists 90°.

    Demonstrates mixed flat / loft construction sharing the same base.
    """
    r_n   = inner_r + t / 2
    T_ch  = [0, 0, 1]
    root_R = ([base_l + r_n, 0, r_n], [base_l + r_n, base_w, r_n])

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, edge_idx=0, angle_deg=90, child_length=arm_h)   # flat arm
    m.extend_loft(base, edge_idx=1, angle_deg=90,
                  free_curve=_free_line(*root_R, T_ch, arm_h, twist_deg=-90))
    return m


def twisted_u_flared(base_l=80, base_w=50, arm_h=50, t=2.0, inner_r=4.0):
    """U-bracket: both arms flare outward — 30° twist and 40 % wider at free end.

    The arms grow from a 50 mm root to a 70 mm free edge while rotating outward,
    creating a bracket that fans open at the top.
    """
    r_n   = inner_r + t / 2
    T_ch  = [0, 0, 1]
    root_L = ([-r_n, 0,        r_n], [-r_n, base_w, r_n])
    root_R = ([base_l + r_n, 0, r_n], [base_l + r_n, base_w, r_n])

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend_loft(base, edge_idx=0, angle_deg=90,
                  free_curve=_free_line(*root_L, T_ch, arm_h,
                                        twist_deg=+30, width_scale=1.4))
    m.extend_loft(base, edge_idx=1, angle_deg=90,
                  free_curve=_free_line(*root_R, T_ch, arm_h,
                                        twist_deg=-30, width_scale=1.4))
    return m


def twisted_u_inward(base_l=80, base_w=50, arm_h=55, t=2.0, inner_r=4.0):
    """U-bracket: both arms twist 90° inward, folding toward each other.

    Left arm free end points in +X; right arm free end points in -X.
    Creates a claw / pincer silhouette when viewed from the front.
    """
    r_n   = inner_r + t / 2
    T_ch  = [0, 0, 1]
    root_L = ([-r_n, 0,        r_n], [-r_n, base_w, r_n])
    root_R = ([base_l + r_n, 0, r_n], [base_l + r_n, base_w, r_n])

    m    = OCC_BracketModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    # Left arm: W_end = rot([0,1,0],[0,0,1],-90°) = [+1,0,0] (inward)
    m.extend_loft(base, edge_idx=0, angle_deg=90,
                  free_curve=_free_line(*root_L, T_ch, arm_h, twist_deg=-90))
    # Right arm: W_end = rot([0,1,0],[0,0,1],+90°) = [-1,0,0] (inward)
    m.extend_loft(base, edge_idx=1, angle_deg=90,
                  free_curve=_free_line(*root_R, T_ch, arm_h, twist_deg=+90))
    return m


# ─── main ─────────────────────────────────────────────────────────────────────

SAMPLES = [
    # (builder_fn, name, title)
    (twisted_l_90,          'twisted_l_90',
     'Twisted L — 90° twist, uniform width'),
    (twisted_l_tapered,     'twisted_l_tapered',
     'Twisted L — 45° twist + taper to 60 % width'),
    (twisted_l_arc_free,    'twisted_l_arc_free',
     'Twisted L — semicircular arc free edge (fan blade)'),
    (twisted_l_bspline_free,'twisted_l_bspline_free',
     'Twisted L — S-curve B-spline free edge'),
    (twisted_u_outward,     'twisted_u_outward',
     'Twisted U — both arms 90° outward'),
    (twisted_u_one_twisted, 'twisted_u_one_twisted',
     'Twisted U — one flat arm, one 90°-twisted arm'),
    (twisted_u_flared,      'twisted_u_flared',
     'Twisted U — both arms 30° flared + 40 % wider'),
    (twisted_u_inward,      'twisted_u_inward',
     'Twisted U — both arms 90° inward (claw/pincer)'),
]


if __name__ == '__main__':
    print(f'Generating {len(SAMPLES)} loft-panel bracket samples…')
    for fn, name, title in SAMPLES:
        m = fn()
        save(m, name, title)
    print('Done →', OUT)
