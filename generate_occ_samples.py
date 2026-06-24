"""
generate_occ_samples.py — pythonOCC backend for all 50 bracket samples.

Run inside the occ312 conda environment:
    conda run -n occ312 python generate_occ_samples.py

Outputs:
  output_occ/<name>.png   — 3-view render (OCC tessellation → matplotlib)
  output_occ/step/<name>.step — STEP AP203 B-Rep file (exact cylindrical arcs)
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# OCC bracket model (drop-in replacement for SheetMetalModel)
from sheet_metal_occ import OCC_BracketModel as SheetMetalModel

# BSplineEdge still comes from the existing package (pure numpy/scipy, no OCC)
from sheet_metal import BSplineEdge

OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_occ')
STEP_OUT = os.path.join(OUT, 'step')
os.makedirs(OUT,      exist_ok=True)
os.makedirs(STEP_OUT, exist_ok=True)

VIEWS = [(28, -50, 'Iso'), (0, 90, 'Side'), (90, -90, 'Top')]


def save(m, name, title):
    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim, label) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        m.render(ax, elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
    fig.suptitle(f'[OCC] {title}', fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    step_path = os.path.join(STEP_OUT, f'{name}.step')
    ok  = m.export_step(step_path)
    tag = 'step✓' if ok else 'step!'
    print(f'  {name}.png  [{tag}]')


# ─── builders (identical to generate_samples.py — API is compatible) ─────────

def l_bracket(angle=90, base_l=100, base_w=60, flange_l=50, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def z_bracket(a1=90, a2=90, base_l=100, web_l=60, flange_l=50,
              base_w=60, t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,    _ = m.extend(base, 1,  a1, web_l,    bend_dir=+1)
    flange, _ = m.extend(web,  1, -a2, flange_l, bend_dir=+1)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def u_bracket(angle=90, base_l=100, base_w=60, arm_l=50, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l, bend_dir=+1)
    m.extend(base, 1, angle, arm_l, bend_dir=+1)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def hat_section(angle=90, base_l=100, base_w=80, web_l=40, flange_l=30,
                t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    left,  _ = m.extend(base,  0,  angle, web_l,    bend_dir=+1)
    right, _ = m.extend(base,  1,  angle, web_l,    bend_dir=+1)
    m.extend(left,  1, -angle, flange_l, bend_dir=+1)
    m.extend(right, 1, -angle, flange_l, bend_dir=+1)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def partial_l(angle=90, sf=0.2, ef=0.8, taper='arc',
              base_l=100, base_w=60, flange_l=50, t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l,
                          start_frac=sf, end_frac=ef, taper=taper)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def partial_z(a1=90, a2=90, sf=0.15, ef=0.75,
              base_l=100, web_l=60, flange_l=50, base_w=60, t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,   _ = m.extend(base, 1,  a1, web_l)
    flange, _ = m.extend(web,  1, -a2, flange_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def partial_u(angle=90, sf=0.2, ef=0.8, base_l=100, base_w=60,
              arm_l=50, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l)
    m.extend(base, 1, angle, arm_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def cross_bracket(angle=90, base_l=80, base_w=60, arm_l=45, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.extend(base, 0, angle, arm_l)
    m.extend(base, 1, angle, arm_l)
    m.extend(base, 2, angle, arm_l)
    m.extend(base, 3, angle, arm_l)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def l_with_side_stiffener(angle=90, base_l=100, base_w=60, flange_l=50,
                           stiff_l=30, sf=0.3, ef=0.7, t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l)
    m.extend(base, 3, angle, stiff_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def deep_chain(angles=(90, -90, 90), lengths=(60, 50, 40, 30), base_w=60,
               t=2.0, inner_r=5.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0,0,0], [1,0,0], [0,1,0], lengths[0], base_w, name='base')
    for a, l in zip(angles, lengths[1:]):
        panel, _ = m.extend(panel, 1, a, l)
    m.mark_hole(m.panels[0], lengths[0]*0.5, base_w*0.5, 'F1')
    return m


def z_with_side_tab(a1=90, a2=90, base_l=100, web_l=60, flange_l=50,
                    tab_l=25, base_w=60, sf=0.2, ef=0.8, t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    web,   _ = m.extend(base, 1,  a1, web_l)
    flange, _ = m.extend(web,  1, -a2, flange_l)
    m.extend(web, 3, 90, tab_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base,   base_l*0.5,   base_w*0.5,        'F1')
    m.mark_hole(flange, flange_l*0.5, flange.width*0.5,  'F2')
    return m


def box_with_tabs(angle=90, base_l=100, base_w=60, arm_l=50, tab_l=20,
                  sf=0.1, ef=0.9, t=2.0, inner_r=5.0):
    m        = SheetMetalModel(t=t, inner_r=inner_r)
    base     = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    left,  _ = m.extend(base, 0, angle, arm_l)
    right, _ = m.extend(base, 1, angle, arm_l)
    m.extend(left,  3, 90, tab_l, start_frac=sf, end_frac=ef)
    m.extend(right, 3, 90, tab_l, start_frac=sf, end_frac=ef)
    m.mark_hole(base, base_l*0.5, base_w*0.5, 'F1')
    return m


def panel_with_rect_recess(base_l=150, base_w=100, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=90, height=60, corner_r=12,
                 r_top=4, r_bot=4, wall_h=6)
    m.mark_hole(base, 20, base_w/2, 'F1')
    m.mark_hole(base, base_l-20, base_w/2, 'F2')
    return m


def panel_with_circ_recess(base_l=150, base_w=100, t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=60, height=60, corner_r=30,
                 r_top=4, r_bot=4, wall_h=5)
    m.mark_hole(base, 20, base_w/2, 'F1')
    m.mark_hole(base, base_l-20, base_w/2, 'F2')
    return m


def l_bracket_with_recess(angle=90, base_l=150, base_w=90, flange_l=60,
                           t=2.0, inner_r=5.0):
    m      = SheetMetalModel(t=t, inner_r=inner_r)
    base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], base_l, base_w, name='base')
    flange, _ = m.extend(base, 1, angle, flange_l)
    m.add_recess(base, cx=base_l/2, cy=base_w/2,
                 width=80, height=50, corner_r=10,
                 r_top=3, r_bot=3, wall_h=5)
    m.mark_hole(base,   20,         base_w/2,         'F1')
    m.mark_hole(flange, flange_l/2, flange.width/2,   'F2')
    return m


def hexagonal_panel(t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    R = 60.0
    hex_verts = [[R*math.cos(i*math.pi/3), R*math.sin(i*math.pi/3)]
                 for i in range(6)]
    min_u = min(v[0] for v in hex_verts)
    min_v = min(v[1] for v in hex_verts)
    hex_verts = [[u - min_u + 5, v - min_v + 5] for u, v in hex_verts]
    base = m.add_polygon_panel([0,0,0], [1,0,0], [0,1,0], hex_verts, name='hex_base')
    flange, _ = m.extend(base, 0, 90, 40)
    cx = sum(v[0] for v in hex_verts) / 6
    cy = sum(v[1] for v in hex_verts) / 6
    tri = [[cx, cy+20], [cx+18, cy-12], [cx-18, cy-12]]
    m.add_polygon_recess(base, tri, corner_r=4.0, r_top=3, r_bot=3, wall_h=4)
    return m


def triangle_panel_chain(t=2.0, inner_r=5.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    tri_verts = [[0, 0], [120, 0], [60, 80.0]]
    base = m.add_polygon_panel([0,0,0], [1,0,0], [0,1,0], tri_verts, name='gusset')
    m.extend(base, 0, 90, 35)
    m.extend(base, 1, 90, 25)
    return m


def polygon_recess_showcase(t=2.0, inner_r=5.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0,0,0], [1,0,0], [0,1,0], 280, 100, name='base')
    tri   = [[45, 20], [90, 80], [0, 80]]
    m.add_polygon_recess(panel, tri, corner_r=6.0, r_top=3, r_bot=3, wall_h=5)
    def poly(n, cx, cy, r, phase=0):
        return [[cx + r*math.cos(2*math.pi*i/n + phase),
                 cy + r*math.sin(2*math.pi*i/n + phase)] for i in range(n)]
    penta = poly(5, 145, 50, 30, phase=math.pi/2)
    m.add_polygon_recess(panel, penta, corner_r=5.0,
                         r_top=3, r_bot=3, wall_h=4, draft_angle=10.0)
    hexa = poly(6, 235, 50, 30)
    m.add_polygon_recess(panel, hexa, corner_r=4.0,
                         r_top=3, r_bot=3, wall_h=5)
    return m


def split_flange(t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 150, 80, name='base')
    m.extend(base, 2, 90, 45, start_frac=0.00, end_frac=0.45)
    m.extend(base, 2, 90, 45, start_frac=0.55, end_frac=1.00)
    return m


def three_equal_tabs(t=2.0, inner_r=4.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 180, 70, name='base')
    gap  = 0.03
    span = (1.0 - 2*gap) / 3
    for k in range(3):
        sf = k * (span + gap)
        m.extend(base, 3, 90, 25, start_frac=sf, end_frac=sf + span)
    return m


def staggered_double_flanges(t=2.0, inner_r=5.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 160, 90,
                        name='base', corner_r=8.0)
    m.extend(base, 2,  90, 40, start_frac=0.00, end_frac=0.48)
    m.extend(base, 2,  60, 40, start_frac=0.52, end_frac=1.00)
    m.extend(base, 1,  90, 20, start_frac=0.10, end_frac=0.45)
    m.extend(base, 1,  90, 20, start_frac=0.55, end_frac=0.90)
    return m


def mid_edge_tab_rounded(t=2.0, inner_r=5.0, corner_r=12.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 140, 80,
                        name='base', corner_r=corner_r)
    m.extend(base, 2, 90, 40, start_frac=0.30, end_frac=0.70)
    return m


def concave_l_panel(t=2.0, inner_r=5.0, corner_r=8.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,40],[60,40],[60,90],[0,90]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='L', corner_r=corner_r)
    m.extend(base, 0, 90, 35)
    m.extend(base, 4, 90, 30)
    return m


def concave_l_inner_tab(t=2.0, inner_r=5.0, corner_r=6.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,40],[60,40],[60,90],[0,90]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='L', corner_r=corner_r)
    m.extend(base, 0, 90, 35)
    m.extend(base, 2, 90, 20)
    return m


def concave_u_panel(t=2.0, inner_r=5.0, corner_r=6.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[140,0],[140,80],[90,80],[90,40],[50,40],[50,80],[0,80]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='U', corner_r=corner_r)
    m.extend(base, 0, 90, 40)
    m.extend(base, 2, 90, 25)
    m.extend(base, 6, 90, 25)
    return m


def concave_t_panel(t=2.0, inner_r=5.0, corner_r=6.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,30],[80,30],[80,90],[40,90],[40,30],[0,30]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='T', corner_r=corner_r)
    m.extend(base, 0, 90, 30)
    m.extend(base, 1, 90, 20)
    m.extend(base, 4, 90, 25)
    m.extend(base, 7, 90, 20)
    return m


def rounded_l_bracket(angle=90, base_l=120, base_w=70, flange_l=50,
                       t=2.0, inner_r=5.0, corner_r=10.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], base_l, base_w,
                        name='base', corner_r=corner_r)
    m.extend(base, 1, angle, flange_l)
    return m


def rounded_base_one_arm(angle=90, base_l=120, base_w=70, arm_l=50,
                          t=2.0, inner_r=5.0, corner_r=8.0):
    m    = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], base_l, base_w,
                        name='base', corner_r=corner_r)
    m.extend(base, 2, angle, arm_l)
    return m


def rounded_hex_panel(t=2.0, inner_r=5.0, corner_r=8.0):
    m = SheetMetalModel(t=t, inner_r=inner_r)
    R = 55.0
    hex_verts = [[R*math.cos(i*math.pi/3) + R + 5,
                  R*math.sin(i*math.pi/3) + R + 5] for i in range(6)]
    base = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0],
                                hex_verts, name='hex', corner_r=corner_r)
    m.extend(base, 0, 90, 35)
    return m


def panel_with_drafted_recess(draft_angle=20.0, t=2.0, inner_r=5.0):
    m     = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0,0,0], [1,0,0], [0,1,0], 200, 120, name='base')
    m.add_recess(panel, cx=65, cy=60, width=80, height=60, corner_r=14,
                 r_top=3, r_bot=3, wall_h=6, draft_angle=draft_angle)
    m.add_recess(panel, cx=155, cy=60, width=50, height=50, corner_r=25,
                 r_top=3, r_bot=3, wall_h=4, draft_angle=draft_angle)
    return m


def spline_wave_panel(t=2.0, inner_r=5.0):
    m       = SheetMetalModel(t=t, inner_r=inner_r)
    crv_bot = BSplineEdge([[0,0],[30,20],[90,-20],[120,0]])
    base    = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,80],[0,80]],
        edge_curves=[crv_bot, None, None, None],
        name='wave',
    )
    m.extend(base, 2, 90, 40)
    return m


def spline_arc_panel(t=2.0, inner_r=5.0):
    m       = SheetMetalModel(t=t, inner_r=inner_r)
    crv_top = BSplineEdge([[120,70],[90,100],[60,105],[30,100],[0,70]])
    base    = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,70],[0,70]],
        edge_curves=[None, None, crv_top, None],
        name='dome',
    )
    m.extend(base, 3, 90, 35)
    return m


def spline_curved_bracket(t=2.0, inner_r=5.0):
    m   = SheetMetalModel(t=t, inner_r=inner_r)
    crv = BSplineEdge([[0,0],[25,-20],[75,-20],[100,0]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[100,0],[100,60],[0,60]],
        edge_curves=[crv, None, None, None],
        name='base',
    )
    m.extend(base, 0, 90, 45)
    return m


def spline_curved_U_bracket(t=2.0, inner_r=5.0):
    m       = SheetMetalModel(t=t, inner_r=inner_r)
    crv_bot = BSplineEdge([[0,0],[30,-18],[70,-18],[100,0]])
    crv_top = BSplineEdge([[100,70],[70,88],[30,88],[0,70]])
    base    = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[100,0],[100,70],[0,70]],
        edge_curves=[crv_bot, None, crv_top, None],
        name='base',
    )
    m.extend(base, 0,  90, 40)
    m.extend(base, 2, -90, 40)
    return m


def spline_s_bracket(t=2.0, inner_r=5.0):
    m   = SheetMetalModel(t=t, inner_r=inner_r)
    crv = BSplineEdge([[0,0],[20,25],[50,-25],[70,25],[100,-15],[120,0]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,75],[0,75]],
        edge_curves=[crv, None, None, None],
        name='base',
    )
    m.extend(base, 0, 90, 50)
    return m


def spline_rounded_spline_panel(t=2.0, inner_r=5.0):
    m       = SheetMetalModel(t=t, inner_r=inner_r)
    crv_bot = BSplineEdge([[0,0],[30,-15],[90,-15],[120,0]])
    crv_top = BSplineEdge([[0,70],[30,85],[90,85],[120,70]])
    base    = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,70],[0,70]],
        edge_curves=[crv_bot, None, crv_top, None],
        name='base', corner_r=10.0,
    )
    m.extend(base, 1, 90, 35)
    m.extend(base, 3, 90, 35)
    return m


# ─── sample table ─────────────────────────────────────────────────────────────

SAMPLES = [
    ('01_L_90',  'L-bracket  90°',  lambda: l_bracket(angle=90)),
    ('02_L_60',  'L-bracket  60°',  lambda: l_bracket(angle=60)),
    ('03_L_120', 'L-bracket 120°',  lambda: l_bracket(angle=120)),
    ('04_L_135', 'L-bracket 135°',  lambda: l_bracket(angle=135)),
    ('05_Z_90_90',  'Z-bracket  90° / 90°',     lambda: z_bracket(a1=90, a2=90)),
    ('06_Z_75_105', 'Z-bracket  75° / 105°',    lambda: z_bracket(a1=75, a2=105)),
    ('07_Z_60_120', 'Z-bracket  60° / 120°',    lambda: z_bracket(a1=60, a2=120)),
    ('08_Z_90_60',  'Z-bracket  90° / 60° asym',lambda: z_bracket(a1=90, a2=60, flange_l=70)),
    ('09_U_90',  'U-bracket  90°',  lambda: u_bracket(angle=90)),
    ('10_U_60',  'U-bracket  60°',  lambda: u_bracket(angle=60)),
    ('11_U_120', 'U-bracket 120°',  lambda: u_bracket(angle=120)),
    ('12_Hat_90', 'Hat section  90°', lambda: hat_section(angle=90)),
    ('13_Hat_75', 'Hat section  75°', lambda: hat_section(angle=75)),
    ('14_PL_center_90',  'Partial-L  30%–70%  arc  90°',   lambda: partial_l(angle=90,  sf=0.30, ef=0.70, taper='arc')),
    ('15_PL_offset_90',  'Partial-L  10%–65%  arc  90°',   lambda: partial_l(angle=90,  sf=0.10, ef=0.65, taper='arc')),
    ('16_PL_linear_90',  'Partial-L  20%–80%  lin  90°',   lambda: partial_l(angle=90,  sf=0.20, ef=0.80, taper='linear')),
    ('17_PL_shallow_60', 'Partial-L  25%–75%  arc  60°',   lambda: partial_l(angle=60,  sf=0.25, ef=0.75, taper='arc')),
    ('18_PL_steep_120',  'Partial-L  20%–80%  arc 120°',   lambda: partial_l(angle=120, sf=0.20, ef=0.80, taper='arc')),
    ('19_PZ_90_90', 'Partial-Z  90°/90°  flange 15%–75%', lambda: partial_z(a1=90, a2=90, sf=0.15, ef=0.75)),
    ('20_PU_90',    'Partial-U  90°  one arm 20%–80%',     lambda: partial_u(angle=90, sf=0.20, ef=0.80)),
    ('21_Cross_90',        'Cross bracket  90°',              lambda: cross_bracket(angle=90)),
    ('22_L_side_stiffener','L + side stiffener  90°',         lambda: l_with_side_stiffener(angle=90)),
    ('23_deep_chain',      '4-panel chain  90° / −90° / 90°', lambda: deep_chain()),
    ('24_Z_side_tab',      'Z + web side tab  90°',           lambda: z_with_side_tab()),
    ('25_box_tabs',        'U + arm side tabs  90°',          lambda: box_with_tabs()),
    ('26_rect_recess',  'Flat panel — rect recess  r4/w6/r4',  lambda: panel_with_rect_recess()),
    ('27_circ_recess',  'Flat panel — circ recess  r4/w5/r4',  lambda: panel_with_circ_recess()),
    ('28_L_recess',     'L-bracket + rect recess  90°',         lambda: l_bracket_with_recess()),
    ('29_drafted_20',   'Flat panel — rect+circ recess  20° draft', lambda: panel_with_drafted_recess(draft_angle=20.0)),
    ('30_drafted_35',   'Flat panel — rect+circ recess  35° draft', lambda: panel_with_drafted_recess(draft_angle=35.0)),
    ('31_hex_panel',    'Hexagonal panel + flange + tri recess',  lambda: hexagonal_panel()),
    ('32_tri_gusset',   'Triangular gusset + two flanges',        lambda: triangle_panel_chain()),
    ('33_poly_recesses','Flat panel — tri/penta/hex recesses',    lambda: polygon_recess_showcase()),
    ('34_rounded_L',    'L-bracket  r=10 mm',           lambda: rounded_l_bracket()),
    ('35_rounded_arm',  'Panel + bottom arm  r=8 mm',   lambda: rounded_base_one_arm()),
    ('36_rounded_hex',  'Hexagonal panel  r=8 mm',      lambda: rounded_hex_panel()),
    ('37_split_flange',          'Base + two partial flanges  gap=10%', lambda: split_flange()),
    ('38_three_tabs',            'Base + three equal tabs',              lambda: three_equal_tabs()),
    ('39_staggered_flanges',     'Two edges split flanges + corner_r',   lambda: staggered_double_flanges()),
    ('40_mid_tab_rounded',       'Mid-edge tab [30%–70%] all corners free', lambda: mid_edge_tab_rounded()),
    ('41_L_concave',     'L-shaped concave panel + 3 flanges',     lambda: concave_l_panel()),
    ('42_L_inner_tab',   'L-shaped + inner-step flange',            lambda: concave_l_inner_tab()),
    ('43_U_concave',     'U-shaped (notch) panel + arm flanges',    lambda: concave_u_panel()),
    ('44_T_concave',     'T-shaped panel + 4 flanges',              lambda: concave_t_panel()),
    ('45_spline_wave',   'SplinePanel S-curve bottom + straight flange',  lambda: spline_wave_panel()),
    ('46_spline_dome',   'SplinePanel dome top + straight flange',         lambda: spline_arc_panel()),
    ('47_spline_L',      'SplinePanel convex bottom → CurvedBendArc 90°',  lambda: spline_curved_bracket()),
    ('48_spline_U',      'SplinePanel two curved edges → two CurvedBendArcs', lambda: spline_curved_U_bracket()),
    ('49_spline_S',      'SplinePanel sinusoidal edge (6-pt B-spline)',     lambda: spline_s_bracket()),
    ('50_spline_rounded','SplinePanel two curved edges + corner_r + 2 flanges', lambda: spline_rounded_spline_panel()),
]


if __name__ == '__main__':
    print(f'Generating {len(SAMPLES)} samples → {OUT}/')
    n_ok = 0
    for name, title, builder in SAMPLES:
        try:
            m = builder()
            save(m, name, title)
            n_ok += 1
        except Exception as exc:
            print(f'  FAILED {name}: {exc}')
    print(f'\nDone.  {n_ok}/{len(SAMPLES)} succeeded.')
