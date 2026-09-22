"""
Three-panel chain bracket samples.

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
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

from sheet_metal.basics.maths import n, rot
from sheet_metal.basics.curve import Curve3D
from sheet_metal.basics.model import SheetMetalModel
from sheet_metal.basics.panel import Panel, FlatPanel
# ── geometry helpers ──────────────────────────────────────────────────────────


def prof(shift_t, shift_n=0.0, shift_w=0.0,
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
    st = spec['shift_t'] # relative
    sn = spec.get('shift_n', 0.0)
    sw = spec.get('shift_w', 0.0)
    w  = spec.get('w')
    width = float(w) if w is not None else root_width

    W_rot = n(rot(_W, _T, spec.get('twist', 0.0)))
    W_rot = n(rot(W_rot, _N, spec.get('twist2', 0.0)))

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

def build_chain_bracket(p_input: dict):
    """Build a chain bracket from parameter dict *p_input*."""

    p = dict(_BASE)
    p.update(p_input)
    
    fastener_panels = []
    bend_r1 = float(p['bend_r1'])
    bend_r2 = float(p['bend_r2'])

    m = SheetMetalModel()
    p1 = m.add_panel('P1', [0, 0, 0], [1, 0, 0], [0, 1, 0],
                     p['p1_l'], p['p1_w'],
                     corner_r=p['corner_r_main'])

    arc1 = m.extend_bend(p1, edge_idx=2, angle_deg=p['mid_ang'], bend_r=bend_r1, child_name='MID')

    keys = ['shift_t', 'shift_n', 'shift_w', 'twist', 'twist2']
    accu_shift_t = {key: 0.0 for key in keys}
    for s in p['mid_profiles']:
        for key in keys:
            s[key] += accu_shift_t[key]
            accu_shift_t[key] = float(s[key])

    curves = [_profile_curve(origin=0.5 * (arc1.p0_out + arc1.p1_out), _T=arc1.T_out, _N=arc1.N_out, _W=arc1.W, root_width=arc1.L, spec=s)
              for s in p['mid_profiles']]

    middle = m.extend_loft_panel('MID', arc1, panel_type='loft',
                               free_curve=curves[-1],
                               intermediate_curves=curves[:-1],
                               ruled=False)
    m.register_in_tree(p1, edge_idx=2, child=middle, arc=arc1, start_frac=0.0, end_frac=1.0)

    p3, _ = m.extend(middle, edge_idx=2,   # CCW free profile edge on LoftPanel
                     angle_deg=p['p3_ang'],
                     panel_type='flat',
                     child_length=p['p3_l'],
                     child_corner_r=p['corner_r_main'],
                     bend_r=bend_r2,
                     bend_dir=np.sign(p['p3_ang']),
                     name='P2')

    if p['sp1_edge'] is not None:
        sp1 = _add_small_panel(m, p1,
                               p['sp1_edge'], p['sp1_ang'], p['sp1_len'],
                               p['sp1_f0'], p['sp1_f1'],
                               p['sp1_blend'], p['sp1_cr'])
        m.mark_hole(sp1, sp1.length / 2.0, sp1.width / 2.0, label='F1')
        fastener_panels.append(np.array([sp1._pt3(*pt) for pt in sp1.vertices]))
    else:
        sp1 = None

    if p['sp3_edge'] is not None:
        sp3 = _add_small_panel(m, p3,
                               p['sp3_edge'], p['sp3_ang'], p['sp3_len'],
                               p['sp3_f0'], p['sp3_f1'],
                               p['sp3_blend'], p['sp3_cr'])
        m.mark_hole(sp3, sp3.length / 2.0, sp3.width / 2.0, label='F3')
        fastener_panels.append(np.array([sp3._pt3(*pt) for pt in sp3.vertices]))
    else:
        sp3 = None

    return m, fastener_panels


# ── sample parameter sets ─────────────────────────────────────────────────────

_BASE = dict(
    bend_r1=50.0, bend_r2=10.0,
    p1_l=80.0, p1_w=50.0,
    mid_ang=90.0,
    mid_profiles=[prof(20), prof(40), prof(60)],
    p3_ang=90.0, p3_l=80.0,
    corner_r_main=8.0,
    sp1_edge=3, sp1_ang=90.0, sp1_len=28.0,
    sp1_f0=0.3, sp1_f1=0.85, sp1_blend=8.0, sp1_cr=0.0,
    sp3_edge=3, sp3_ang=90.0, sp3_len=28.0,
    sp3_f0=0.15, sp3_f1=0.85, sp3_blend=8.0, sp3_cr=0.0,
)