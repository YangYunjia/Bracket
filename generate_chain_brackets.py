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

Middle loft: root profile fixed by B1 arc; three movable profiles placed
via _prof() (shift along T_mid / W / lateral, dual twists, width).  The
leading edge (t=0) of every profile is pinned to a straight guide
a0_root + shift_t·T_mid so the loft long edge stays aligned with the
bend arcs.  Lateral bow (shift_lat) offsets the trailing edge only.
Skinned with smooth ThruSections (ruled=False).

Bend radii: inner_r_b1 (P1→middle), inner_r_b2 (middle→P3); fallback inner_r.
"""

# ── Shape parameters — edit these to change the generated geometry ────────────
#
T_THICK       = 2.0
INNER_R       = 4.0
INNER_R_B1    = None
INNER_R_B2    = None
#
P1_L          = 80.0
P1_W          = 50.0
#
MID_ANG       = 90.0
MID_PROFILES  = None   # list of 3 _prof() dicts; see _BASE for default
#
P3_ANG        = 90.0
P3_L          = 80.0
CORNER_R_MAIN = 8.0
#
SP1_EDGE, SP1_ANG, SP1_LEN = 3, 90.0, 28.0
SP1_F0, SP1_F1, SP1_BLEND, SP1_CR = 0.15, 0.85, 8.0, 5.0
SP3_EDGE, SP3_ANG, SP3_LEN = 3, 90.0, 28.0
SP3_F0, SP3_F1, SP3_BLEND, SP3_CR = 0.15, 0.85, 8.0, 5.0
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sheet_metal.rendering.occ_model import OCC_Renderer
from sheet_metal.basics.panel import Curve3D
from sheet_metal.basics.model import SheetMetalModel

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
    v, k = np.asarray(v, float), _n(np.asarray(k, float))
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1.0 - np.cos(t))


def _prof(shift_t, shift_y=0.0, shift_lat=0.0,
          twist=0.0, twist2=0.0, w=None):
    """One loft cross-section spec.

    shift_t    : distance along T_mid from root leading corner (mm)
    shift_y    : slide entire profile along W_root (mm)
    shift_lat  : lateral bow — offsets trailing edge along lat_dir (mm)
    twist      : roll around T_mid (degrees)
    twist2     : yaw around lat_dir (degrees)
    w          : profile width mm; None = root width
    """
    return dict(shift_t=float(shift_t), shift_y=float(shift_y),
                shift_lat=float(shift_lat),
                twist=float(twist), twist2=float(twist2), w=w)


def _root_frame(p1_l, p1_w, mid_ang, r_n):
    """Root leading corner, trailing corner, T_mid, W_root, lat_dir."""
    alpha = np.radians(abs(mid_ang))
    root_x = p1_l + r_n * np.sin(alpha)
    root_z = r_n * (1.0 - np.cos(alpha))
    root_left  = np.array([root_x, 0.0,       root_z])
    root_right = np.array([root_x, p1_w,      root_z])
    T_mid  = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    W_root = _n(root_right - root_left)
    lat_dir = _n(np.cross(T_mid, W_root))
    return root_left, root_right, T_mid, W_root, lat_dir


def _profile_curve(a0_root, W_root, T_mid, lat_dir, root_width, spec):
    """Build one Curve3D; leading corner pinned to the T_mid guide."""
    st = spec['shift_t']
    sy = spec.get('shift_y', 0.0)
    sl = spec.get('shift_lat', 0.0)
    w  = spec.get('w')
    width = float(w) if w is not None else root_width

    W_rot = _n(_rot(W_root, T_mid, spec.get('twist', 0.0)))
    W_rot = _n(_rot(W_rot, lat_dir, spec.get('twist2', 0.0)))

    a0 = a0_root + st * T_mid + sy * W_root
    b0 = a0 + width * W_rot + sl * lat_dir
    return Curve3D.line(a0, b0)


def _loft_profiles(a0_root, root_width, W_root, T_mid, lat_dir, profile_specs):
    """Build intermediate + free profiles from three explicit specs."""
    if len(profile_specs) != 3:
        raise ValueError('mid_profiles must contain exactly 3 entries.')
    curves = [_profile_curve(a0_root, W_root, T_mid, lat_dir, root_width, s)
              for s in profile_specs]
    return curves[2], curves[:2]


def _add_small_panel(m, parent, edge_idx, angle_deg, arm_len,
                     f0, f1, blend_r, corner_r):
    sp, _ = m.extend(parent, edge_idx, angle_deg, arm_len,
                     start_frac=f0, end_frac=f1, blend_r=blend_r)
    sp.corner_r = float(corner_r)
    sp._register_child(0, 0.0, 1.0)
    return sp


# ── main bracket builder ──────────────────────────────────────────────────────

def build_chain_bracket(p):
    """Build a chain bracket from parameter dict *p*."""
    thick = float(p['thick'])
    r_in  = float(p['inner_r'])
    r_in_b1 = float(p.get('inner_r_b1', r_in))
    r_in_b2 = float(p.get('inner_r_b2', r_in))
    r_n_b1  = r_in_b1 + thick / 2.0

    m = SheetMetalModel(t=thick, inner_r=r_in)
    p1 = m.add_panel([0, 0, 0], [1, 0, 0], [0, 1, 0],
                     p['p1_l'], p['p1_w'],
                     name='P1', corner_r=p['corner_r_main'])

    a0_root, b0_root, T_mid, W_root, lat_dir = _root_frame(
        p['p1_l'], p['p1_w'], p['mid_ang'], r_n_b1)

    free_curve, intermediates = _loft_profiles(
        a0_root, p['p1_w'], W_root, T_mid, lat_dir, p['mid_profiles'])

    middle, _ = m.extend_arm(p1, edge_idx=1, panel_type='loft',
                               angle_deg=p['mid_ang'],
                               free_curve=free_curve,
                               intermediate_curves=intermediates,
                               ruled=False,
                               bend_r=r_in_b1)

    p3, _ = m.extend(middle, edge_idx=2,   # CCW free profile edge on LoftPanel
                     angle_deg=p['p3_ang'],
                     child_length=p['p3_l'],
                     child_corner_r=p['corner_r_main'],
                     bend_r=r_in_b2)

    sp1 = _add_small_panel(m, p1,
                           p['sp1_edge'], p['sp1_ang'], p['sp1_len'],
                           p['sp1_f0'], p['sp1_f1'],
                           p['sp1_blend'], p['sp1_cr'])

    sp3 = _add_small_panel(m, p3,
                           p['sp3_edge'], p['sp3_ang'], p['sp3_len'],
                           p['sp3_f0'], p['sp3_f1'],
                           p['sp3_blend'], p['sp3_cr'])

    # m.mark_hole(sp1, sp1.length / 2.0, sp1.width / 2.0, label='F1')
    # m.mark_hole(sp3, sp3.length / 2.0, sp3.width / 2.0, label='F3')

    return m


# ── sample parameter sets ─────────────────────────────────────────────────────

_BASE = dict(
    thick=2.0, inner_r=4.0,
    inner_r_b1=4.0, inner_r_b2=4.0,
    p1_l=80.0, p1_w=50.0,
    mid_ang=90.0,
    mid_profiles=[_prof(20), _prof(40), _prof(60)],
    p3_ang=90.0, p3_l=80.0,
    corner_r_main=8.0,
    sp1_edge=3, sp1_ang=90.0, sp1_len=28.0,
    sp1_f0=0.3, sp1_f1=0.85, sp1_blend=8.0, sp1_cr=5.0,
    sp3_edge=3, sp3_ang=90.0, sp3_len=28.0,
    sp3_f0=0.15, sp3_f1=0.85, sp3_blend=8.0, sp3_cr=5.0,
)


def _s(**kw):
    d = dict(_BASE)
    d.update(kw)
    return d

SAMPLES1 = [
    ('chain_01_basic',
     'C-curve middle — 15 mm lateral bow at mid profile',
     _s(mid_profiles=[
         _prof(22),
         _prof(43, shift_lat=15),
         _prof(100)],
        mid_ang=30.0,
        sp1_edge=0, sp1_f0=0.05, sp1_f1=0.8,
        sp3_edge=1, sp3_f0=0.1, sp3_f1=0.9)),
]

SAMPLES = [
    ('chain_01_basic',
     'Basic symmetric — B1=B2=90°, straight middle',
     _s()),

    ('chain_02_tilted',
     'Tilted angles — B1=70°, B2=110°, wide P1, short P3',
     _s(mid_ang=70.0, p3_ang=110.0,
        mid_profiles=[_prof(23), _prof(47), _prof(70)],
        p1_l=90.0, p3_l=65.0,
        sp1_edge=2, sp1_ang=85.0, sp1_f0=0.1, sp1_f1=0.9,
        sp3_edge=1, sp3_ang=85.0, sp3_f0=0.2, sp3_f1=0.75)),

    ('chain_03_twisted_30',
     'Twisted middle — 30° roll at free edge',
     _s(mid_profiles=[
         _prof(22),
         _prof(43, twist=10),
         _prof(65, twist=30)],
        sp1_edge=3, sp3_edge=3)),

    ('chain_04_c_curve',
     'C-curve middle — 15 mm lateral bow at mid profile',
     _s(mid_profiles=[
         _prof(22),
         _prof(43, shift_lat=15),
         _prof(65)],
        sp1_edge=0, sp1_f0=0.1, sp1_f1=0.9,
        sp3_edge=1, sp3_f0=0.1, sp3_f1=0.9)),

    ('chain_05_twisted_c_curve',
     'Twisted + lateral bow — 25° roll, 12 mm lateral at P2',
     _s(mid_profiles=[
         _prof(23, twist=8),
         _prof(47, shift_lat=12, twist=17),
         _prof(70, twist=25)],
        sp1_edge=2, sp1_f0=0.2, sp1_f1=0.75,
        sp3_edge=2, sp3_f0=0.25, sp3_f1=0.8)),

    ('chain_06_asymmetric',
     'Asymmetric — wide P1, B1=80°, 15° twist, short P3',
     _s(p1_l=100.0, p1_w=60.0, p3_l=55.0,
        mid_ang=80.0,
        mid_profiles=[_prof(27), _prof(53, twist=8), _prof(80, twist=15)],
        corner_r_main=10.0,
        sp1_ang=80.0, sp1_len=32.0,
        sp3_ang=95.0, sp3_len=22.0)),

    ('chain_07_shallow',
     'Shallow bends — B1=B2=45°',
     _s(mid_ang=45.0, p3_ang=45.0,
        mid_profiles=[_prof(27), _prof(53), _prof(80)],
        p1_l=90.0, p3_l=90.0,
        sp1_f0=0.2, sp1_f1=0.75,
        sp3_f0=0.25, sp3_f1=0.8)),

    ('chain_08_steep_twist',
     'Steep + twist — B1=120°, 40° roll, B2=80°',
     _s(mid_ang=120.0, p3_ang=80.0, p3_l=70.0,
        mid_profiles=[
            _prof(20, twist=13),
            _prof(40, twist=27),
            _prof(60, twist=40)],
        sp1_edge=0, sp1_ang=80.0, sp1_f0=0.1, sp1_f1=0.85,
        sp3_edge=1, sp3_ang=80.0, sp3_f0=0.1, sp3_f1=0.85)),

    ('chain_09_large_panels',
     'Large panels, C-curve at mid profile',
     _s(p1_l=110.0, p1_w=65.0, p3_l=110.0,
        mid_profiles=[_prof(25), _prof(50, shift_lat=12), _prof(75)],
        corner_r_main=12.0,
        sp1_len=38.0, sp1_f0=0.2, sp1_f1=0.8,
        sp3_len=38.0, sp3_f0=0.2, sp3_f1=0.8)),

    ('chain_10_complex',
     'Complex — mixed twists, lateral bow, width taper',
     _s(p1_l=85.0, p1_w=55.0, p3_l=75.0,
        mid_ang=85.0, p3_ang=95.0, corner_r_main=9.0,
        mid_profiles=[
            _prof(25, twist=12),
            _prof(50, shift_lat=10, twist=24, twist2=5),
            _prof(75, shift_y=5, twist=35, twist2=8, w=52)],
        sp1_edge=2, sp1_ang=85.0, sp1_len=30.0,
        sp1_f0=0.2, sp1_f1=0.75, sp1_blend=9.0, sp1_cr=6.0,
        sp3_edge=3, sp3_ang=95.0, sp3_len=25.0,
        sp3_f0=0.25, sp3_f1=0.8, sp3_blend=7.0, sp3_cr=6.0)),

    ('chain_11_narrow_middle',
     'Width taper — P1 W=58, middle/P3 W=32',
     _s(p1_w=58.0,
        mid_profiles=[_prof(23), _prof(47), _prof(70, w=32)],
        sp1_edge=2, sp3_edge=2)),

    ('chain_12_flared_middle',
     'Flared middle — P1 W=40, middle/P3 W=62',
     _s(p1_w=40.0,
        mid_profiles=[_prof(22, w=46), _prof(43, w=54), _prof(65, w=62)],
        sp1_edge=0, sp1_f0=0.1, sp1_f1=0.9,
        sp3_edge=1, sp3_f0=0.1, sp3_f1=0.9)),

    ('chain_13_y_shift',
     'Y-shifted loft — 18 mm slide at free edge',
     _s(p1_w=52.0,
        mid_profiles=[_prof(23), _prof(45, shift_y=9), _prof(68, shift_y=18)],
        sp1_edge=3, sp3_edge=3)),

    ('chain_14_dual_twist',
     'Dual-axis twist — 25° roll + 20° yaw at free',
     _s(mid_profiles=[
         _prof(23, twist=8, twist2=7),
         _prof(47, twist=17, twist2=13),
         _prof(70, twist=25, twist2=20)],
        sp1_edge=2, sp3_edge=2)),

    ('chain_15_y_shift_narrow',
     'Y-shift + width taper — P1 W=55, mid/P3 W=38',
     _s(p1_w=55.0,
        mid_profiles=[
            _prof(24, w=48),
            _prof(48, shift_y=6, w=42),
            _prof(72, shift_y=12, w=38)],
        sp1_edge=0, sp3_edge=1)),

    ('chain_16_opposing_twist',
     'Opposing twists — +30° roll, −25° yaw',
     _s(mid_profiles=[
         _prof(22, twist=10, twist2=-8),
         _prof(43, twist=20, twist2=-17),
         _prof(65, twist=30, twist2=-25)],
        p1_l=85.0, p3_l=75.0,
        sp1_edge=3, sp3_edge=3)),

    ('chain_17_three_widths',
     'Three widths — P1=65, mid/P3=42',
     _s(p1_w=65.0, p1_l=95.0, p3_l=60.0,
        mid_ang=80.0,
        mid_profiles=[_prof(25, w=55), _prof(50, w=48), _prof(75, w=42)],
        sp1_len=32.0, sp3_len=26.0)),

    ('chain_18_y_dual_twist',
     'Y-shift + dual twist at each profile',
     _s(mid_profiles=[
         _prof(23, shift_y=5, twist=7, twist2=5),
         _prof(47, shift_y=10, twist=14, twist2=10),
         _prof(70, shift_y=15, twist=20, twist2=15)],
        sp1_edge=2, sp1_ang=85.0,
        sp3_edge=2, sp3_ang=85.0)),

    ('chain_19_extreme_taper',
     'Extreme taper + Y-shift — P1 W=70, mid/P3 W=28',
     _s(p1_w=70.0, mid_ang=100.0, p3_ang=85.0,
        mid_profiles=[
            _prof(27, w=55, shift_y=7),
            _prof(53, w=40, shift_y=15),
            _prof(80, w=28, shift_y=22)],
        sp1_edge=0, sp1_f0=0.15, sp1_f1=0.85,
        sp3_edge=1, sp3_f0=0.2, sp3_f1=0.8)),

    ('chain_20_full_freedom',
     'Full freedom — width, Y, roll, yaw, lateral',
     _s(p1_w=50.0, mid_ang=85.0, p3_ang=95.0,
        mid_profiles=[
            _prof(25, twist=8, twist2=-6),
            _prof(50, shift_lat=8, shift_y=5, twist=15, twist2=-12, w=54),
            _prof(75, shift_y=10, twist=22, twist2=-18, w=58)],
        sp1_edge=2, sp1_ang=90.0, sp1_f0=0.2, sp1_f1=0.75,
        sp3_edge=3, sp3_ang=90.0, sp3_f0=0.25, sp3_f1=0.8)),

    ('chain_21_blend_b1_tight',
     'Tight B1 blend — inner_r_b1=2 mm, B2=6 mm',
     _s(inner_r_b1=2.0, inner_r_b2=6.0,
        mid_profiles=[_prof(23), _prof(47), _prof(70)])),

    ('chain_22_blend_b2_tight',
     'Tight B2 blend — inner_r_b1=8 mm, B2=2.5 mm',
     _s(inner_r_b1=8.0, inner_r_b2=2.5, p3_ang=80.0,
        mid_profiles=[_prof(23), _prof(47), _prof(70)])),

    ('chain_23_blend_asymmetric',
     'Asymmetric blends + shaped middle — B1=3, B2=10 mm',
     _s(inner_r_b1=3.0, inner_r_b2=10.0,
        mid_ang=75.0, p3_ang=100.0,
        mid_profiles=[
            _prof(22, shift_lat=6),
            _prof(45, shift_lat=14, twist=18),
            _prof(68, shift_y=8, twist=28, w=55)])),
]


if __name__ == '__main__':
    sample_set = SAMPLES1
    print(f'Generating {len(sample_set)} chain bracket samples…')
    renderer = OCC_Renderer()

    for name, title, params in sample_set:
        m = build_chain_bracket(params)
        groups, markers = renderer.render(m)
        fig = plt.figure(figsize=(14, 5))

        for i, (elev, azim, label) in enumerate(VIEWS, 1):
            ax = fig.add_subplot(1, 3, i, projection='3d')
            renderer.export_plot(groups, markers, ax=ax, elev=elev, azim=azim)
            ax.set_title(label, fontsize=9)
        fig.suptitle(f'[ChainBracket] {title}', fontsize=10)
        plt.tight_layout()
        path = os.path.join(OUT, f'{name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        step_path = os.path.join(STEP_OUT, f'{name}.step')
        ok  = renderer.export_step(groups, step_path)
        tag = 'step✓' if ok else 'step!'
        print(f'  {name}.png  [{tag}]')

    print('Done →', OUT)
