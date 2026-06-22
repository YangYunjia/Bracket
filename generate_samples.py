"""
generate_samples.py — produce 20 bracket samples with varied settings.

Each sample is saved as a 3-view PNG (isometric / side / top) in output/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sheet_metal import SheetMetalModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT, exist_ok=True)

VIEWS = [(28, -50, 'Iso'), (0, 90, 'Side'), (90, -90, 'Top')]


def save(m, name, title):
    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim, label) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        m.render(ax, elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}.png')


# ─── builders ────────────────────────────────────────────────────────────────

def l_bracket(angle=90, base_l=100, base_w=60, flange_l=50, t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l)
    m.mark_hole(base,   base_l*0.5, base_w*0.5, 'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def z_bracket(a1=90, a2=90, base_l=100, web_l=60, flange_l=50,
              base_w=60, t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,    _ = m.extend(base, 1,  a1, web_l,    bend_dir=+1)
    flange, _ = m.extend(web,  1, -a2, flange_l, bend_dir=+1)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,     'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def u_bracket(angle=90, base_l=100, base_w=60, arm_l=50, t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l, bend_dir=+1)
    m.extend(base, 1, angle, arm_l, bend_dir=+1)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def hat_section(angle=90, base_l=100, base_w=80, web_l=40, flange_l=30,
                t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base  = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    left,  _ = m.extend(base,  0,  angle, web_l,    bend_dir=+1)
    right, _ = m.extend(base,  1,  angle, web_l,    bend_dir=+1)
    m.extend(left,  1, -angle, flange_l, bend_dir=+1)
    m.extend(right, 1, -angle, flange_l, bend_dir=+1)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def partial_l(angle=90, sf=0.2, ef=0.8, taper='arc',
              base_l=100, base_w=60, flange_l=50, t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l,
                          start_frac=sf, end_frac=ef, taper=taper)
    m.mark_hole(base,   base_l*0.5, base_w*0.5, 'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def partial_z(a1=90, a2=90, sf=0.15, ef=0.75,
              base_l=100, web_l=60, flange_l=50, base_w=60,
              t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,   _ = m.extend(base, 1,  a1, web_l)
    flange, _ = m.extend(web,  1, -a2, flange_l,
                          start_frac=sf, end_frac=ef)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,      'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def partial_u(angle=90, sf=0.2, ef=0.8, base_l=100, base_w=60,
              arm_l=50, t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l)
    m.extend(base, 1, angle, arm_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


# ── side-edge and multi-level builders ───────────────────────────────────────

def cross_bracket(angle=90, base_l=80, base_w=60, arm_l=45, t=2.0, inner_r=5.0):
    """Plus shape: one arm from each of the 4 base edges."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l)   # e0: −T end
    m.extend(base, 1, angle, arm_l)   # e1: +T end
    m.extend(base, 2, angle, arm_l)   # e2: −W side
    m.extend(base, 3, angle, arm_l)   # e3: +W side
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def l_with_side_stiffener(angle=90, base_l=100, base_w=60, flange_l=50,
                           stiff_l=30, sf=0.3, ef=0.7, t=2.0, inner_r=5.0):
    """L-bracket (e1) with a partial stiffener rib from the side edge (e3)."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle,  flange_l)
    m.extend(base, 3, angle, stiff_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,      'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def deep_chain(angles=(90, -90, 90), lengths=(60, 50, 40, 30),
               base_w=60, t=2.0, inner_r=5.0):
    """4-panel chain: base → A → B → C, each bent from the e1 of the previous."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0,0,0], [1,0,0], [0,1,0], lengths[0], base_w, name='base')
    for a, l in zip(angles, lengths[1:]):
        panel, _ = m.extend(panel, 1, a, l)
    m.mark_hole(m.panels[0], lengths[0]*0.5, base_w*0.5, 'F1')
    return m


def z_with_side_tab(a1=90, a2=90, base_l=100, web_l=60, flange_l=50,
                    tab_l=25, base_w=60, sf=0.2, ef=0.8, t=2.0, inner_r=5.0):
    """Z-bracket plus a partial tab from the side edge (e3) of the web."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,   _ = m.extend(base, 1,  a1, web_l)
    flange, _ = m.extend(web,  1, -a2, flange_l)
    m.extend(web, 3, 90, tab_l, start_frac=sf, end_frac=ef)   # tab off web side
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,      'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5, 'F2')
    return m


def box_with_tabs(angle=90, base_l=100, base_w=60, arm_l=50, tab_l=20,
                  sf=0.1, ef=0.9, t=2.0, inner_r=5.0):
    """U-bracket with partial tabs folded from the +W sides of both arms."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base     = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    left,  _ = m.extend(base, 0, angle, arm_l)
    right, _ = m.extend(base, 1, angle, arm_l)
    m.extend(left,  3, 90, tab_l, start_frac=sf, end_frac=ef)
    m.extend(right, 3, 90, tab_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


# ── stamped recess builders ───────────────────────────────────────────────────

def panel_with_rect_recess(base_l=150, base_w=100, t=2.0, inner_r=5.0):
    """Flat panel with a single rounded-rectangle recess."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=90, height=60, corner_r=12,
                 r_top=4, r_bot=4, wall_h=6)
    m.mark_hole(base, 20, base_w/2, 'F1')
    m.mark_hole(base, base_l-20, base_w/2, 'F2')
    return m


def panel_with_circ_recess(base_l=150, base_w=100, t=2.0, inner_r=5.0):
    """Flat panel with a circular recess (corner_r = width/2)."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=60, height=60, corner_r=30,   # corner_r = width/2 → circle
                 r_top=4, r_bot=4, wall_h=5)
    m.mark_hole(base, 20, base_w/2, 'F1')
    m.mark_hole(base, base_l-20, base_w/2, 'F2')
    return m


def l_bracket_with_recess(angle=90, base_l=150, base_w=90, flange_l=60,
                           t=2.0, inner_r=5.0):
    """L-bracket with a drawn recess on the base panel."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l)
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=80, height=50, corner_r=10,
                 r_top=3, r_bot=3, wall_h=5)
    m.mark_hole(base, 20, base_w/2, 'F1')
    m.mark_hole(flange, flange_l/2, flange.width/2, 'F2')
    return m


# ─── sample table ─────────────────────────────────────────────────────────────

SAMPLES = [
    # (filename, title, builder_call)

    # L-brackets — varying angle
    ('01_L_90',  'L-bracket  90°',
     lambda: l_bracket(angle=90)),
    ('02_L_60',  'L-bracket  60°',
     lambda: l_bracket(angle=60)),
    ('03_L_120', 'L-bracket 120°',
     lambda: l_bracket(angle=120)),
    ('04_L_135', 'L-bracket 135°',
     lambda: l_bracket(angle=135)),

    # Z-brackets — varying angle pairs
    ('05_Z_90_90',   'Z-bracket  90° / 90°',
     lambda: z_bracket(a1=90,  a2=90)),
    ('06_Z_75_105',  'Z-bracket  75° / 105°',
     lambda: z_bracket(a1=75,  a2=105)),
    ('07_Z_60_120',  'Z-bracket  60° / 120°',
     lambda: z_bracket(a1=60,  a2=120)),
    ('08_Z_90_60',   'Z-bracket  90° / 60°  (asymmetric flange)',
     lambda: z_bracket(a1=90,  a2=60, flange_l=70)),

    # U-brackets
    ('09_U_90',  'U-bracket  90°',
     lambda: u_bracket(angle=90)),
    ('10_U_60',  'U-bracket  60°',
     lambda: u_bracket(angle=60)),
    ('11_U_120', 'U-bracket 120°',
     lambda: u_bracket(angle=120)),

    # Hat sections
    ('12_Hat_90', 'Hat section  90°',
     lambda: hat_section(angle=90)),
    ('13_Hat_75', 'Hat section  75°',
     lambda: hat_section(angle=75)),

    # Partial-flange L-brackets — varying coverage
    ('14_PL_center_90',  'Partial-L  30%–70%  arc taper  90°',
     lambda: partial_l(angle=90,  sf=0.30, ef=0.70, taper='arc')),
    ('15_PL_offset_90',  'Partial-L  10%–65%  arc taper  90°',
     lambda: partial_l(angle=90,  sf=0.10, ef=0.65, taper='arc')),
    ('16_PL_linear_90',  'Partial-L  20%–80%  linear taper  90°',
     lambda: partial_l(angle=90,  sf=0.20, ef=0.80, taper='linear')),
    ('17_PL_shallow_60', 'Partial-L  25%–75%  arc taper  60°',
     lambda: partial_l(angle=60,  sf=0.25, ef=0.75, taper='arc')),
    ('18_PL_steep_120',  'Partial-L  20%–80%  arc taper 120°',
     lambda: partial_l(angle=120, sf=0.20, ef=0.80, taper='arc')),

    # Partial-flange Z and U
    ('19_PZ_90_90', 'Partial-Z  90° / 90°  flange 15%–75%',
     lambda: partial_z(a1=90, a2=90, sf=0.15, ef=0.75)),
    ('20_PU_90',    'Partial-U  90°  one arm 20%–80%',
     lambda: partial_u(angle=90, sf=0.20, ef=0.80)),

    # Side-edge and multi-level extensions
    ('21_Cross_90',        'Cross bracket  90°  (all 4 edges)',
     lambda: cross_bracket(angle=90)),
    ('22_L_side_stiffener','L-bracket + side stiffener  90°',
     lambda: l_with_side_stiffener(angle=90)),
    ('23_deep_chain',      '4-panel chain  90° / −90° / 90°',
     lambda: deep_chain()),
    ('24_Z_side_tab',      'Z-bracket + web side tab  90°',
     lambda: z_with_side_tab()),
    ('25_box_tabs',        'U-bracket + arm side tabs  90°',
     lambda: box_with_tabs()),

    # Stamped recesses
    ('26_rect_recess',   'Flat panel — rounded-rect recess  r4/w6/r4',
     lambda: panel_with_rect_recess()),
    ('27_circ_recess',   'Flat panel — circular recess  r4/w5/r4',
     lambda: panel_with_circ_recess()),
    ('28_L_recess',      'L-bracket + rect recess on base  90°',
     lambda: l_bracket_with_recess()),
]


if __name__ == '__main__':
    print(f'Generating {len(SAMPLES)} samples → {OUT}/')
    for name, title, builder in SAMPLES:
        m = builder()
        save(m, name, title)
        m.print_tree()
        print()
    print('Done.')
