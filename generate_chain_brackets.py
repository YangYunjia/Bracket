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
from sheet_metal.basics.curve import Curve3D
from sheet_metal.basics.model import SheetMetalModel
from sheet_metal.basics.panel import Panel, FlatPanel

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


def _prof(shift_t, shift_n=0.0, shift_w=0.0,
          twist=0.0, twist2=0.0, w=None):
    """One loft cross-section spec.

    shift_t    : distance along T_mid from root leading corner (mm)
    shift_y    : slide entire profile along W_root (mm)
    shift_lat  : lateral bow — offsets trailing edge along lat_dir (mm)
    twist      : roll around T_mid (degrees)
    twist2     : yaw around lat_dir (degrees)
    w          : profile width mm; None = root width
    """
    return dict(shift_t=float(shift_t), shift_n=float(shift_n), shift_w=float(shift_w),
                twist=float(twist), twist2=float(twist2), w=w)


def _profile_curve(origin, _T, _N, _W, root_width, spec):
    """Build one Curve3D; leading corner pinned to the T_mid guide."""
    st = spec['shift_t']
    sn = spec.get('shift_n', 0.0)
    sw = spec.get('shift_w', 0.0)
    w  = spec.get('w')
    width = float(w) if w is not None else root_width

    W_rot = _n(_rot(_W, _T, spec.get('twist', 0.0)))
    W_rot = _n(_rot(W_rot, _N, spec.get('twist2', 0.0)))

    origin0 = origin + st * _T + sn * _N
    a0 = origin0 + (sw - 0.5 * width) * W_rot
    b0 = origin0 + (sw + 0.5 * width) * W_rot
    return Curve3D.line(a0, b0)


def _add_small_panel(m: SheetMetalModel, parent: Panel, edge_idx: int, angle_deg: float, arm_len: float,
                     f0, f1, blend_r, corner_r, name=None) -> FlatPanel:
    if name is None:
        name = f'{parent.name}.e{edge_idx}.SP'
    sp, _ = m.extend(parent, edge_idx, angle_deg, panel_type='flat',
                     child_length=arm_len, child_corner_r=corner_r,
                     start_frac=f0, end_frac=f1, blend_r=blend_r, name=name)
    return sp


# ── main bracket builder ──────────────────────────────────────────────────────

def build_chain_bracket(p):
    """Build a chain bracket from parameter dict *p*."""
    bend_r1 = float(p['bend_r1'])
    bend_r2 = float(p['bend_r2'])

    m = SheetMetalModel()
    p1 = m.add_panel('P1', [0, 0, 0], [1, 0, 0], [0, 1, 0],
                     p['p1_l'], p['p1_w'],
                     corner_r=p['corner_r_main'])

    arc1 = m.extend_bend(p1, edge_idx=1, angle_deg=p['mid_ang'], bend_r=bend_r1, child_name='MID')

    curves = [_profile_curve(origin=0.5 * (arc1.p0_out + arc1.p1_out), _T=arc1.T_out, _N=arc1.N_out, _W=arc1.W, root_width=arc1.L, spec=s)
              for s in p['mid_profiles']]

    middle = m.extend_loft_panel('MID', arc1, panel_type='loft',
                               free_curve=curves[-1],
                               intermediate_curves=curves[:-1],
                               ruled=False)
    m.register_in_tree(p1, edge_idx=1, child=middle, arc=arc1, start_frac=0.0, end_frac=1.0)

    p3, _ = m.extend(middle, edge_idx=2,   # CCW free profile edge on LoftPanel
                     angle_deg=p['p3_ang'],
                     panel_type='flat',
                     child_length=p['p3_l'],
                     child_corner_r=p['corner_r_main'],
                     bend_r=bend_r2,
                     name='P2')

    if p['sp1_edge'] is not None:
        sp1 = _add_small_panel(m, p1,
                               p['sp1_edge'], p['sp1_ang'], p['sp1_len'],
                               p['sp1_f0'], p['sp1_f1'],
                               p['sp1_blend'], p['sp1_cr'])
        m.mark_hole(sp1, sp1.length / 2.0, sp1.width / 2.0, label='F1')
    else:
        sp1 = None

    if p['sp3_edge'] is not None:
        sp3 = _add_small_panel(m, p3,
                               p['sp3_edge'], p['sp3_ang'], p['sp3_len'],
                               p['sp3_f0'], p['sp3_f1'],
                               p['sp3_blend'], p['sp3_cr'])
        m.mark_hole(sp3, sp3.length / 2.0, sp3.width / 2.0, label='F3')
    else:
        sp3 = None

    return m


# ── sample parameter sets ─────────────────────────────────────────────────────

_BASE = dict(
    bend_r1=50.0, bend_r2=10.0,
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
    ('00_only_backbone',
     '',
     _s(p1_w=100, mid_profiles=[
         _prof(100),
         _prof(250),
         _prof(400)],
        mid_ang=30.0,
        sp1_edge=None, sp3_edge=None)),
    ('chain_01_basic',
     'C-curve middle — 15 mm lateral bow at mid profile',
     _s(p1_w=100, mid_profiles=[
         _prof(100),
         _prof(250, shift_n=15, twist=10),
         _prof(400, twist2=10)],
        mid_ang=30.0,
        sp1_edge=0, sp1_f0=0.05, sp1_f1=0.8,
        sp3_edge=1, sp3_f0=0.4, sp3_f1=0.6)),
]

if __name__ == '__main__':
    sample_set = SAMPLES1
    print(f'Generating {len(sample_set)} chain bracket samples…')
    renderer = OCC_Renderer()
    tessellateds = {}

    for name, title, params in sample_set:
        m = build_chain_bracket(params)
        groups, markers = renderer.render(m)
        tessellated = renderer.export_tessellated(groups)
        tessellateds[name] = tessellated
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

        step_path = os.path.join(STEP_OUT, f'{name}')
        ok  = renderer.export_step(groups, step_path + '.step')
        ok  = renderer.export_stl(groups, step_path + '.stl')
        tag = 'step✓' if ok else 'step!'
        print(f'  {name}.png  [{tag}]')

    # save the tessellated dictionaries to a single file
    np.savez(os.path.join(OUT, 'tessellated.npz'), tessellateds)

    print('Done →', OUT)
