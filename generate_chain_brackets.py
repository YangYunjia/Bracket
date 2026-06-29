"""
generate_chain_brackets.py — Three-panel chain bracket samples.

Run in the occ312 conda environment:
    conda run -n occ312 python generate_chain_brackets.py

Outputs:
  output_chain/<name>.png       — 3-view render (ISO / Side / Top)
  output_chain/step/<name>.step — STEP AP203 B-Rep file

Chain bracket topology
──────────────────────
  P1 (flat) ──[arc B1]── Middle (loft) ──[arc B2]── P3 (flat)
  SP1 ──[partial arc]──┘                             └──[partial arc]── SP3
  fastener mark at centre of SP1 and SP3

All three main panels share the same width edge.  P1 and P3 are flat
rectangles with rounded free corners.  Middle is a smooth loft through
4 cross-section profiles (root + 2 intermediates + free) with optional
width-axis twist and a lateral S/C-curve offset.
"""

# ── Shape parameters — edit these to change the generated geometry ────────────
#
# Material
T_THICK       = 2.0    # sheet thickness (mm)
INNER_R       = 4.0    # inner bend radius (mm)
#
# Panel 1 — first flat panel, origin at (0,0,0), T=[1,0,0], W=[0,1,0]
P1_L          = 80.0   # length along T (mm)
P1_W          = 50.0   # width along W (mm); also sets the middle-panel width
#
# Middle loft panel
MID_ANG       = 90.0   # bend angle at P1→middle junction (degrees)
MID_H         = 60.0   # loft height along T_mid (mm)
MID_TWIST     = 0.0    # rotation of width axis at free profile around T_mid (degrees)
MID_LAT       = 0.0    # lateral C-curve offset at profile mid-point (mm);
                        # creates a banana bow perpendicular to T_mid and W_root
MID_RULED     = False  # True = ruled surface, False = smooth loft
#
# Panel 3 — second flat panel grown from middle's free edge
P3_ANG        = 90.0   # bend angle at middle→P3 junction (degrees)
P3_L          = 80.0   # length of P3 (mm); width is determined by middle free edge
#
# Corner rounding for main panels (free corners only — root corners are left sharp)
CORNER_R_MAIN = 8.0    # mm
#
# Small panel on P1 (connected via PartialBendArc)
SP1_EDGE      = 3      # P1 edge: 0=left, 2=bottom, 3=top  (NOT 1 = junction to middle)
SP1_ANG       = 90.0   # bend angle (degrees)
SP1_LEN       = 28.0   # arm length (mm)
SP1_F0        = 0.15   # start fraction along the chosen P1 edge
SP1_F1        = 0.85   # end   fraction
SP1_BLEND     = 8.0    # partial-arc blend radius (mm)
SP1_CR        = 5.0    # corner rounding for SP1 free corners (mm)
#
# Small panel on P3
SP3_EDGE      = 3      # P3 edge: 1=far-end, 2=bottom, 3=top  (NOT 0 = junction to middle)
SP3_ANG       = 90.0
SP3_LEN       = 28.0
SP3_F0        = 0.15
SP3_F1        = 0.85
SP3_BLEND     = 8.0
SP3_CR        = 5.0
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sheet_metal_occ import OCC_BracketModel
from sheet_metal import Curve3D

OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_chain')
STEP_OUT = os.path.join(OUT, 'step')
os.makedirs(OUT,      exist_ok=True)
os.makedirs(STEP_OUT, exist_ok=True)

VIEWS = [(28, -50, 'ISO'), (0, 90, 'Side'), (90, -90, 'Top')]


# ── geometry helpers ──────────────────────────────────────────────────────────

def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v if m < 1e-12 else v / m


def _rot(v, k, deg):
    """Rodrigues rotation of vector v around unit axis k by deg degrees."""
    v, k = np.asarray(v, float), _n(np.asarray(k, float))
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1.0 - np.cos(t))


def _mid_profiles(root_left, root_right, T_mid, H, twist_deg, lat):
    """Build 2 intermediate Curve3D profiles and 1 free Curve3D for the middle loft.

    The 4 profiles are: [root (auto by extend_loft), interp1, interp2, free].
    Intermediate profiles sit at 1/3 and 2/3 of the loft height; the free
    profile is at 1.0 (no lateral offset, so p3 attaches cleanly).
    Twist rotates the width axis linearly from 0 at root to twist_deg at free.
    Lateral offset creates a C-curve bow using sin(frac·π) (zero at root and free).

    Returns (free_curve, [interp1, interp2]).
    """
    root_left  = np.asarray(root_left, float)
    root_right = np.asarray(root_right, float)
    T_mid      = _n(np.asarray(T_mid, float))
    mid_root   = (root_left + root_right) / 2.0
    half_w     = np.linalg.norm(root_right - root_left) / 2.0
    W_root     = _n(root_right - root_left)

    # Lateral direction: perpendicular to T_mid within the T_mid–W_root plane
    lat_dir = _n(np.cross(T_mid, W_root))

    curves = []
    for frac in (1.0 / 3.0, 2.0 / 3.0, 1.0):
        center  = mid_root + frac * H * T_mid
        if abs(lat) > 1e-9:
            # C-curve bow: peaks at mid-height, zero at root and free
            center = center + lat * np.sin(frac * np.pi) * lat_dir
        W_rot = _n(_rot(W_root, T_mid, twist_deg * frac))
        curves.append(Curve3D.line(center - half_w * W_rot,
                                   center + half_w * W_rot))

    return curves[2], curves[:2]   # (free_curve, [interp1, interp2])


def _add_small_panel(m, parent, edge_idx, angle_deg, arm_len,
                     f0, f1, blend_r, corner_r):
    """Extend a small flat panel with PartialBendArc and rounded free corners.

    The root corners (at u=0 of the child) are blocked from rounding by
    registering edge 0 as occupied, so only the far-end corners are rounded.
    """
    sp, _ = m.extend(parent, edge_idx, angle_deg, arm_len,
                     start_frac=f0, end_frac=f1, blend_r=blend_r)
    sp.corner_r = float(corner_r)
    # Block root corners (edge 0 of child is always the junction to parent)
    sp._register_child(0, 0.0, 1.0)
    return sp


# ── main bracket builder ──────────────────────────────────────────────────────

def build_chain_bracket(p):
    """Build a chain bracket from parameter dict p.

    Required keys: thick, inner_r, p1_l, p1_w, mid_ang, mid_h, mid_twist,
    mid_lat, mid_ruled, p3_ang, p3_l, corner_r_main,
    sp1_edge, sp1_ang, sp1_len, sp1_f0, sp1_f1, sp1_blend, sp1_cr,
    sp3_edge, sp3_ang, sp3_len, sp3_f0, sp3_f1, sp3_blend, sp3_cr.
    """
    thick = float(p['thick'])
    r_in  = float(p['inner_r'])
    r_n   = r_in + thick / 2.0   # neutral-axis bend radius

    m = OCC_BracketModel(t=thick, inner_r=r_in)

    # ── Panel 1 ───────────────────────────────────────────────────────────────
    p1 = m.add_panel([0, 0, 0], [1, 0, 0], [0, 1, 0],
                     p['p1_l'], p['p1_w'],
                     name='P1', corner_r=p['corner_r_main'])
    # Note: after extend_loft registers middle on p1's edge 1, the corners
    # adjacent to edge 1 (C1 and C2) become blocked, leaving only C0 and C3
    # (the left-side free corners) rounded.

    # ── Middle loft panel ─────────────────────────────────────────────────────
    # Compute root endpoints using the same formula as extend_loft().
    # P1 has N=[0,0,1], T=[1,0,0], and middle connects via P1's edge 1
    # (T_out=[1,0,0], bdir=[0,0,1]).
    alpha1   = np.radians(abs(p['mid_ang']))
    root_x   = p['p1_l'] + r_n * np.sin(alpha1)
    root_z   = r_n * (1.0 - np.cos(alpha1))
    root_left  = np.array([root_x, 0.0,         root_z])
    root_right = np.array([root_x, p['p1_w'],   root_z])
    # T_mid = sin(alpha)*[0,0,1] + cos(alpha)*[1,0,0] = [cos(alpha), 0, sin(alpha)]
    T_mid = np.array([np.cos(alpha1), 0.0, np.sin(alpha1)])

    free_curve, intermediates = _mid_profiles(
        root_left, root_right, T_mid,
        p['mid_h'], p.get('mid_twist', 0.0), p.get('mid_lat', 0.0))

    middle, _ = m.extend_loft(p1, edge_idx=1,
                               angle_deg=p['mid_ang'],
                               free_curve=free_curve,
                               intermediate_curves=intermediates,
                               ruled=p.get('mid_ruled', False))

    # ── Panel 3 ───────────────────────────────────────────────────────────────
    p3, _ = m.extend(middle, edge_idx=1,
                     angle_deg=p['p3_ang'],
                     child_length=p['p3_l'])
    p3.corner_r = p['corner_r_main']
    # P3's root edge is edge 0; register it as occupied to block root corners.
    p3._register_child(0, 0.0, 1.0)

    # ── Small panel on P1 ─────────────────────────────────────────────────────
    sp1 = _add_small_panel(m, p1,
                           p['sp1_edge'], p['sp1_ang'], p['sp1_len'],
                           p['sp1_f0'],  p['sp1_f1'],
                           p['sp1_blend'], p['sp1_cr'])

    # ── Small panel on P3 ─────────────────────────────────────────────────────
    sp3 = _add_small_panel(m, p3,
                           p['sp3_edge'], p['sp3_ang'], p['sp3_len'],
                           p['sp3_f0'],  p['sp3_f1'],
                           p['sp3_blend'], p['sp3_cr'])

    # ── Fastener marks at centres of small panels ─────────────────────────────
    m.mark_hole(sp1, sp1.length / 2.0, sp1.width / 2.0, label='F1')
    m.mark_hole(sp3, sp3.length / 2.0, sp3.width / 2.0, label='F3')

    return m


# ── render / save helper ──────────────────────────────────────────────────────

def save(m, name, title):
    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim, label) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        m.render(ax, elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
    fig.suptitle(f'[ChainBracket] {title}', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUT, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    step_path = os.path.join(STEP_OUT, f'{name}.step')
    ok  = m.export_step(step_path)
    tag = 'step✓' if ok else 'step!'
    print(f'  {name}.png  [{tag}]')


# ── 10 sample parameter sets ──────────────────────────────────────────────────

_BASE = dict(
    thick=2.0, inner_r=4.0,
    p1_l=80.0,   p1_w=50.0,
    mid_ang=90.0, mid_h=60.0, mid_twist=0.0, mid_lat=0.0, mid_ruled=False,
    p3_ang=90.0,  p3_l=80.0,
    corner_r_main=8.0,
    sp1_edge=3,  sp1_ang=90.0, sp1_len=28.0,
    sp1_f0=0.15, sp1_f1=0.85, sp1_blend=8.0, sp1_cr=5.0,
    sp3_edge=3,  sp3_ang=90.0, sp3_len=28.0,
    sp3_f0=0.15, sp3_f1=0.85, sp3_blend=8.0, sp3_cr=5.0,
)


def _s(**kw):
    """Return a parameter dict by merging BASE with overrides."""
    d = dict(_BASE)
    d.update(kw)
    return d


SAMPLES = [
    # (filename_stem, title, params)
    ('chain_01_basic',
     'Basic symmetric — B1=B2=90°, straight middle',
     _s()),

    # ('chain_02_tilted',
    #  'Tilted angles — B1=70°, B2=110°, wide P1, short P3',
    #  _s(mid_ang=70.0, p3_ang=110.0, mid_h=70.0,
    #     p1_l=90.0, p3_l=65.0,
    #     sp1_edge=2, sp1_ang=85.0, sp1_f0=0.1, sp1_f1=0.9,
    #     sp3_edge=1, sp3_ang=85.0, sp3_f0=0.2, sp3_f1=0.75)),

    # ('chain_03_twisted_30',
    #  'Twisted middle — 30° width twist',
    #  _s(mid_twist=30.0, mid_h=65.0,
    #     sp1_edge=3, sp3_edge=3)),

    # ('chain_04_c_curve',
    #  'C-curve middle — 15 mm lateral bow',
    #  _s(mid_lat=15.0, mid_h=65.0,
    #     sp1_edge=0, sp1_ang=90.0, sp1_f0=0.1, sp1_f1=0.9,
    #     sp3_edge=1, sp3_ang=90.0, sp3_f0=0.1, sp3_f1=0.9)),

    # ('chain_05_twisted_c_curve',
    #  'Twisted + C-curve — 25° twist, 12 mm lateral bow',
    #  _s(mid_twist=25.0, mid_lat=12.0, mid_h=70.0,
    #     sp1_edge=2, sp1_ang=90.0, sp1_f0=0.2, sp1_f1=0.75,
    #     sp3_edge=2, sp3_ang=90.0, sp3_f0=0.25, sp3_f1=0.8)),

    # ('chain_06_asymmetric',
    #  'Asymmetric — wide P1 (W=60), B1=80°, 15° twist, short P3',
    #  _s(p1_l=100.0, p1_w=60.0, p3_l=55.0,
    #     mid_ang=80.0, mid_h=80.0, mid_twist=15.0,
    #     p3_ang=90.0, corner_r_main=10.0,
    #     sp1_ang=80.0, sp1_len=32.0,
    #     sp3_ang=95.0, sp3_len=22.0)),

    # ('chain_07_shallow',
    #  'Shallow bends — B1=B2=45°',
    #  _s(mid_ang=45.0, p3_ang=45.0, mid_h=80.0,
    #     p1_l=90.0, p3_l=90.0,
    #     sp1_f0=0.2, sp1_f1=0.75,
    #     sp3_f0=0.25, sp3_f1=0.8)),

    # ('chain_08_steep_twist',
    #  'Steep + twist — B1=120°, 40° twist, B2=80°',
    #  _s(mid_ang=120.0, mid_h=60.0, mid_twist=40.0,
    #     p3_ang=80.0, p3_l=70.0,
    #     sp1_edge=0, sp1_ang=80.0, sp1_f0=0.1, sp1_f1=0.85,
    #     sp3_edge=1, sp3_ang=80.0, sp3_f0=0.1, sp3_f1=0.85)),

    # ('chain_09_large_panels',
    #  'Large panels, big corner radii, C-curve middle',
    #  _s(p1_l=110.0, p1_w=65.0, p3_l=110.0,
    #     mid_h=75.0, mid_lat=12.0,
    #     corner_r_main=12.0,
    #     sp1_len=38.0, sp1_f0=0.2, sp1_f1=0.8,
    #     sp3_len=38.0, sp3_f0=0.2, sp3_f1=0.8)),

    # ('chain_10_complex',
    #  'Complex — B1=85°, B2=95°, 35° twist, 10 mm bow, mixed SP edges',
    #  _s(p1_l=85.0, p1_w=55.0, p3_l=75.0,
    #     mid_ang=85.0, mid_h=75.0, mid_twist=35.0, mid_lat=10.0,
    #     p3_ang=95.0, corner_r_main=9.0,
    #     sp1_edge=2, sp1_ang=85.0, sp1_len=30.0,
    #     sp1_f0=0.2, sp1_f1=0.75, sp1_blend=9.0, sp1_cr=6.0,
    #     sp3_edge=3, sp3_ang=95.0, sp3_len=25.0,
    #     sp3_f0=0.25, sp3_f1=0.8, sp3_blend=7.0, sp3_cr=6.0)),
]


if __name__ == '__main__':
    print(f'Generating {len(SAMPLES)} chain bracket samples…')
    for name, title, params in SAMPLES:
        try:
            m = build_chain_bracket(params)
            # save(m, name, title)
        except Exception as e:
            import traceback
            print(f'  ERROR {name}: {e}')
            traceback.print_exc()
    print('Done →', OUT)
