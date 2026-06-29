"""
generate_samples.py — produce 20 bracket samples with varied settings.

Each sample is saved as a 3-view PNG (isometric / side / top) in output/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
from sheet_metal import BSplineEdge, SplinePanel
from sheet_metal.rendering.numpy_model import NumpyBracketModel as SheetMetalModel

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


def hexagonal_panel(t=2.0, inner_r=5.0):
    """Hexagonal panel with a flange on one edge and a polygon recess."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    R = 60.0   # circumradius of hexagon
    hex_verts = [[R*math.cos(i*math.pi/3), R*math.sin(i*math.pi/3)]
                 for i in range(6)]
    # Shift so all coords are positive
    min_u = min(v[0] for v in hex_verts)
    min_v = min(v[1] for v in hex_verts)
    hex_verts = [[u - min_u + 5, v - min_v + 5] for u, v in hex_verts]

    base = m.add_polygon_panel([0, 0, 0], [1, 0, 0], [0, 1, 0],
                                hex_verts, name='hex_base')
    # Extend a flange from edge 0 (bottom edge)
    flange, _ = m.extend(base, 0, 90, 40)
    # Triangular polygon recess near the centre of the hex panel
    cx = sum(v[0] for v in hex_verts) / 6
    cy = sum(v[1] for v in hex_verts) / 6
    tri = [[cx, cy+20], [cx+18, cy-12], [cx-18, cy-12]]
    m.add_polygon_recess(base, tri, corner_r=4.0,
                         r_top=3, r_bot=3, wall_h=4)
    return m


def triangle_panel_chain(t=2.0, inner_r=5.0):
    """Triangle gusset with two child flanges on different edges."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    h = 80.0
    tri_verts = [[0, 0], [120, 0], [60, h]]
    base = m.add_polygon_panel([0, 0, 0], [1, 0, 0], [0, 1, 0],
                                tri_verts, name='gusset')
    m.extend(base, 0, 90, 35)           # flange on bottom edge
    m.extend(base, 1, 90, 25)           # flange on right edge
    return m


def polygon_recess_showcase(t=2.0, inner_r=5.0):
    """Flat panel with three different polygon recesses: tri, penta, hex."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0, 0, 0], [1, 0, 0], [0, 1, 0], 280, 100, name='base')

    # Triangular recess (left)
    tri = [[45, 20], [90, 80], [0, 80]]
    m.add_polygon_recess(panel, tri, corner_r=6.0,
                         r_top=3, r_bot=3, wall_h=5)

    # Pentagonal recess (centre)
    def poly(n, cx, cy, r, phase=0):
        return [[cx + r*math.cos(2*math.pi*i/n + phase),
                 cy + r*math.sin(2*math.pi*i/n + phase)] for i in range(n)]
    penta = poly(5, 145, 50, 30, phase=math.pi/2)
    m.add_polygon_recess(panel, penta, corner_r=5.0,
                         r_top=3, r_bot=3, wall_h=4, draft_angle=10.0)

    # Hexagonal recess (right)
    hexa = poly(6, 235, 50, 30)
    m.add_polygon_recess(panel, hexa, corner_r=4.0,
                         r_top=3, r_bot=3, wall_h=5, draft_angle=0.0)

    return m


def split_flange(t=2.0, inner_r=5.0):
    """Base panel with two partial 90° flanges on the bottom edge, gap in middle."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 150, 80, name='base')
    # e2 (bottom): two flanges each 45% wide, 10% gap
    m.extend(base, 2, 90, 45, start_frac=0.00, end_frac=0.45)
    m.extend(base, 2, 90, 45, start_frac=0.55, end_frac=1.00)
    return m


def three_equal_tabs(t=2.0, inner_r=4.0):
    """Base panel with three equal tabs evenly spaced along the top edge."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 180, 70, name='base')
    gap = 0.03          # 3% gap between tabs → ~5.4 mm gap on 180 mm edge
    span = (1.0 - 2*gap) / 3
    for k in range(3):
        sf = k * (span + gap)
        m.extend(base, 3, 90, 25, start_frac=sf, end_frac=sf + span)
    return m


def staggered_double_flanges(t=2.0, inner_r=5.0):
    """Two edges each split into two partial flanges at different angles,
    demonstrating per-edge span tracking and corner blocking interactions."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 160, 90, name='base', corner_r=8.0)
    # bottom edge e2: left half 90°, right half 60°
    m.extend(base, 2,  90, 40, start_frac=0.00, end_frac=0.48)
    m.extend(base, 2,  60, 40, start_frac=0.52, end_frac=1.00)
    # right edge e1: two short tabs
    m.extend(base, 1, 90, 20, start_frac=0.10, end_frac=0.45)
    m.extend(base, 1, 90, 20, start_frac=0.55, end_frac=0.90)
    return m


def mid_edge_tab_rounded(t=2.0, inner_r=5.0, corner_r=12.0):
    """Panel with a mid-edge-only child: child spans 30%-70%, never touching
    either endpoint, so all four corners remain free and get rounded."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], 140, 80, name='base',
                        corner_r=corner_r)
    # e2 bottom, 30–70% only — does NOT reach frac=0 or frac=1
    m.extend(base, 2, 90, 40, start_frac=0.30, end_frac=0.70)
    return m


def concave_l_panel(t=2.0, inner_r=5.0, corner_r=8.0):
    """L-shaped polygon panel.

    CCW vertices: (0,0)→(120,0)→(120,40)→(60,40)→(60,90)→(0,90)
    Concave re-entrant corner at (60,40) — skipped by rounding.
    Five convex corners: the two that are free (not touched by flanges) are rounded.

    Edge map (polygon edge i goes from vertex i to vertex (i+1)%6):
      e0 (0,0)→(120,0)   bottom          outward: -W
      e1 (120,0)→(120,40) right-low      outward: +T
      e2 (120,40)→(60,40) inner h-step   outward: +W  (faces into cutout)
      e3 (60,40)→(60,90)  inner v-step   outward: +T  (faces into cutout)
      e4 (60,90)→(0,90)   top-left       outward: +W
      e5 (0,90)→(0,0)     left           outward: -T
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,40],[60,40],[60,90],[0,90]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='L', corner_r=corner_r)
    m.extend(base, 0, 90, 35)   # bottom full flange (blocks v0, v1)
    m.extend(base, 4, 90, 30)   # top-left flange   (blocks v4, v5)
    # v2=(120,40) is free: both adj. edges (e1, e2) unused → rounded ✓
    # v3=(60,40) is the concave corner → stays sharp regardless
    return m


def concave_l_inner_tab(t=2.0, inner_r=5.0, corner_r=6.0):
    """L-shaped panel with a tab on the inner step edge (e2).

    Demonstrates extending from a concave polygon edge — the flange grows
    into the cutout area, creating an internal shelf feature.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,40],[60,40],[60,90],[0,90]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='L', corner_r=corner_r)
    m.extend(base, 0, 90, 35)          # bottom flange
    # e2: inner horizontal step (120,40)→(60,40), outward = +W (north into cutout)
    m.extend(base, 2, 90, 20)          # inner-step shelf
    return m


def concave_u_panel(t=2.0, inner_r=5.0, corner_r=6.0):
    """U-shaped (notch) polygon panel.

    CCW vertices: (0,0)→(140,0)→(140,80)→(90,80)→(90,40)→(50,40)→(50,80)→(0,80)
    Concave corners at (90,40) and (50,40) — the notch-base re-entrants.
    Flanges on the bottom and both arm tops.

    Edge map:
      e0 (0,0)→(140,0)    bottom              outward: -W
      e1 (140,0)→(140,80) right               outward: +T
      e2 (140,80)→(90,80) top of right arm    outward: +W
      e3 (90,80)→(90,40)  inner right wall    outward: +T
      e4 (90,40)→(50,40)  inner notch base    outward: +W (faces into notch)
      e5 (50,40)→(50,80)  inner left wall     outward: -T
      e6 (50,80)→(0,80)   top of left arm     outward: +W
      e7 (0,80)→(0,0)     left                outward: -T
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[140,0],[140,80],[90,80],[90,40],[50,40],[50,80],[0,80]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='U', corner_r=corner_r)
    m.extend(base, 0, 90, 40)   # bottom full flange
    m.extend(base, 2, 90, 25)   # top of right arm
    m.extend(base, 6, 90, 25)   # top of left arm
    return m


def concave_t_panel(t=2.0, inner_r=5.0, corner_r=6.0):
    """T-shaped polygon panel (crossbar + central stem).

    CCW vertices: (0,0)→(120,0)→(120,30)→(80,30)→(80,90)→(40,90)→(40,30)→(0,30)
    Concave corners at (80,30) and (40,30) — where the stem meets the crossbar.
    Flanges on bottom, stem top, and both crossbar ends.

    Edge map:
      e0 (0,0)→(120,0)   bottom of crossbar   outward: -W
      e1 (120,0)→(120,30) right end            outward: +T
      e2 (120,30)→(80,30) inner step right     outward: +W (into shoulder gap)
      e3 (80,30)→(80,90)  right side of stem   outward: +T
      e4 (80,90)→(40,90)  top of stem          outward: +W
      e5 (40,90)→(40,30)  left side of stem    outward: -T
      e6 (40,30)→(0,30)   inner step left      outward: +W (into shoulder gap)
      e7 (0,30)→(0,0)     left end             outward: -T
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    verts = [[0,0],[120,0],[120,30],[80,30],[80,90],[40,90],[40,30],[0,30]]
    base  = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0], verts,
                                 name='T', corner_r=corner_r)
    m.extend(base, 0, 90, 30)   # bottom of crossbar
    m.extend(base, 1, 90, 20)   # right end of crossbar
    m.extend(base, 4, 90, 25)   # top of stem
    m.extend(base, 7, 90, 20)   # left end of crossbar
    return m


def rounded_l_bracket(angle=90, base_l=120, base_w=70, flange_l=50,
                       t=2.0, inner_r=5.0, corner_r=10.0):
    """L-bracket with rounded free corners.

    base extends from e1 (right edge):
      e1 used → C1 (e1,e2) and C2 (e1,e3) blocked.
      C0 (e0,e2) and C3 (e3,e0) are free → rounded.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], base_l, base_w,
                        name='base', corner_r=corner_r)
    m.extend(base, 1, angle, flange_l)
    return m


def rounded_base_one_arm(angle=90, base_l=120, base_w=70, arm_l=50,
                          t=2.0, inner_r=5.0, corner_r=8.0):
    """Panel with one arm from the bottom edge — top two corners are free.

    e2 (bottom) used → C0 (e0,e2) and C1 (e1,e2) blocked.
    C2 (e1,e3) and C3 (e3,e0) are free → rounded.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    base = m.add_panel([0,0,0],[1,0,0],[0,1,0], base_l, base_w,
                        name='base', corner_r=corner_r)
    m.extend(base, 2, angle, arm_l)
    return m


def rounded_hex_panel(t=2.0, inner_r=5.0, corner_r=8.0):
    """Hexagonal panel with one flange — 4 free hex corners get rounded."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    R = 55.0
    hex_verts = [[R*math.cos(i*math.pi/3) + R + 5,
                  R*math.sin(i*math.pi/3) + R + 5] for i in range(6)]
    base = m.add_polygon_panel([0,0,0],[1,0,0],[0,1,0],
                                hex_verts, name='hex', corner_r=corner_r)
    m.extend(base, 0, 90, 35)   # extend from edge 0
    # Edge 0 connects vertices 0→1, so vertices 0 and 1 are blocked.
    # The other 4 vertices are free → their corners get rounded.
    return m


# ─── spline panel builders ────────────────────────────────────────────────────

def spline_wave_panel(t=2.0, inner_r=5.0):
    """SplinePanel whose bottom edge is a gentle S-curve; straight-edge child.

    The base panel is 120×80 with the bottom edge (e0: from (0,0) to (120,0))
    replaced by an S-shaped B-spline.  One straight edge gets a 90° flange.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    # S-curve edge: from (0,0) up to midpoint then back down
    crv_bot = BSplineEdge([[0,0],[30,20],[90,-20],[120,0]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,80],[0,80]],
        edge_curves=[crv_bot, None, None, None],
        name='wave',
    )
    m.extend(base, 2, 90, 40)   # right edge (straight) → FlatPanel child
    return m


def spline_arc_panel(t=2.0, inner_r=5.0):
    """SplinePanel whose top edge approximates a convex arc; 90° flange on left.

    A 5-control-point cubic B-spline approximates a circular arc on the top
    edge (e2: from (120,70) to (0,70) in the CCW polygon), creating a
    dome-like outline.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    # Top edge arcs up ~30 units at the midpoint
    crv_top = BSplineEdge([[120,70],[90,100],[60,105],[30,100],[0,70]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,70],[0,70]],
        edge_curves=[None, None, crv_top, None],
        name='dome',
    )
    m.extend(base, 3, 90, 35)   # left edge (straight)
    return m


def spline_curved_bracket(t=2.0, inner_r=5.0):
    """SplinePanel with a curved bottom edge; 90° CurvedBendArc + SplinePanel child.

    The parent panel (100×60) has a convex-curved bottom edge (outward bulge
    of ~20 mm at the midpoint).  Extending from this curved edge creates a
    CurvedBendArc and a SplinePanel child whose root edge mirrors the curve.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    crv = BSplineEdge([[0,0],[25,-20],[75,-20],[100,0]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[100,0],[100,60],[0,60]],
        edge_curves=[crv, None, None, None],   # edge 0 is curved
        name='base',
    )
    m.extend(base, 0, 90, 45)   # curved edge → CurvedBendArc + SplinePanel child
    return m


def spline_curved_U_bracket(t=2.0, inner_r=5.0):
    """SplinePanel with two mirrored curved edges → two CurvedBendArcs.

    The base panel (100×70) has both the bottom edge (e0) and the top edge (e2)
    curved outward symmetrically.  Two 90° flanges are extended from those
    curved edges, each generating a CurvedBendArc and a SplinePanel child.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    crv_bot = BSplineEdge([[0,0],[30,-18],[70,-18],[100,0]])
    crv_top = BSplineEdge([[100,70],[70,88],[30,88],[0,70]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[100,0],[100,70],[0,70]],
        edge_curves=[crv_bot, None, crv_top, None],
        name='base',
    )
    m.extend(base, 0, 90, 40)    # bottom curved → first arm
    m.extend(base, 2, -90, 40)   # top curved → second arm (bends in -N direction)
    return m


def spline_s_bracket(t=2.0, inner_r=5.0):
    """SplinePanel with a sinusoidal bottom edge + 90° CurvedBendArc.

    More pronounced S-curve (3 inflection points, 6 control points) to
    demonstrate higher-order B-spline geometry in the arc sweep.
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
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
    """SplinePanel: rectangular with two curved edges + corner_r on straight corners.

    Bottom edge is a convex arc; top edge is a convex arc in the opposite sense.
    Left and right edges are straight, with corner_r=10 on the four straight
    corners (all four corners are at straight-edge junctions).
    """
    m = SheetMetalModel(t=t, inner_r=inner_r)
    crv_bot = BSplineEdge([[0,0],[30,-15],[90,-15],[120,0]])
    crv_top = BSplineEdge([[0,70],[30,85],[90,85],[120,70]])
    base = m.add_spline_panel(
        origin=[0,0,0], T=[1,0,0], W=[0,1,0],
        vertices=[[0,0],[120,0],[120,70],[0,70]],
        edge_curves=[crv_bot, None, crv_top, None],
        name='base', corner_r=10.0,
    )
    # Extend from straight left and right edges (no curves involved)
    m.extend(base, 1, 90, 35)   # right straight edge
    m.extend(base, 3, 90, 35)   # left straight edge
    return m


def panel_with_drafted_recess(draft_angle=20.0, t=2.0, inner_r=5.0):
    """Flat panel showing a rect and a circular recess, both with drafted walls."""
    m = SheetMetalModel(t=t, inner_r=inner_r)
    panel = m.add_panel([0, 0, 0], [1, 0, 0], [0, 1, 0], 200, 120, name='base')
    # Drafted rectangular recess — 20° draft
    m.add_recess(panel, cx=65, cy=60, width=80, height=60, corner_r=14,
                 r_top=3, r_bot=3, wall_h=6, draft_angle=draft_angle)
    # Drafted circular recess (corner_r = width/2) — same draft
    m.add_recess(panel, cx=155, cy=60, width=50, height=50, corner_r=25,
                 r_top=3, r_bot=3, wall_h=4, draft_angle=draft_angle)
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

    # Drafted sidewall recesses
    ('29_drafted_recess_20', 'Flat panel — rect + circ recess  20° draft wall',
     lambda: panel_with_drafted_recess(draft_angle=20.0)),
    ('30_drafted_recess_35', 'Flat panel — rect + circ recess  35° draft wall',
     lambda: panel_with_drafted_recess(draft_angle=35.0)),

    # Polygon panels and recesses
    ('31_hex_panel',          'Hexagonal panel + flange + triangular recess',
     lambda: hexagonal_panel()),
    ('32_triangle_gusset',    'Triangular gusset panel + two flanges',
     lambda: triangle_panel_chain()),
    ('33_polygon_recesses',   'Flat panel — triangle / pentagon / hexagon recesses',
     lambda: polygon_recess_showcase()),

    # Concave polygon panels
    ('41_L_concave',       'L-shaped panel — 1 concave corner sharp, 3 flanges',
     lambda: concave_l_panel()),
    ('42_L_inner_tab',     'L-shaped panel — flange on inner step edge (into cutout)',
     lambda: concave_l_inner_tab()),
    ('43_U_concave',       'U-shaped (notch) panel — 2 concave corners, arm flanges',
     lambda: concave_u_panel()),
    ('44_T_concave',       'T-shaped panel — 2 concave corners at stem base, 4 flanges',
     lambda: concave_t_panel()),

    # Spline panels — Phase 1 (curved outlines, straight-edge arcs)
    ('45_spline_wave',        'SplinePanel S-curved bottom edge + straight 90° flange',
     lambda: spline_wave_panel()),
    ('46_spline_arc_dome',    'SplinePanel dome top edge + straight 90° flange',
     lambda: spline_arc_panel()),

    # Spline panels — Phase 2 (CurvedBendArc + SplinePanel child)
    ('47_spline_curved_L',    'SplinePanel convex bottom edge → CurvedBendArc 90° L-bracket',
     lambda: spline_curved_bracket()),
    ('48_spline_curved_U',    'SplinePanel two curved edges → two CurvedBendArcs 90° U-bracket',
     lambda: spline_curved_U_bracket()),
    ('49_spline_S_bracket',   'SplinePanel sinusoidal edge (6-pt B-spline) → CurvedBendArc',
     lambda: spline_s_bracket()),
    ('50_spline_rounded',     'SplinePanel two curved edges + corner_r on straight corners + 2 flanges',
     lambda: spline_rounded_spline_panel()),

    # Multi-child per edge
    ('37_split_flange',           'Base + two partial 90° flanges on bottom edge  gap=10%',
     lambda: split_flange()),
    ('38_three_equal_tabs',       'Base + three equal 90° tabs along top edge',
     lambda: three_equal_tabs()),
    ('39_staggered_double_flanges','Two edges each split: bottom 90°/60°, right two tabs  r=8 mm',
     lambda: staggered_double_flanges()),
    ('40_mid_edge_tab_rounded',   'Mid-edge-only tab [30%–70%]: all four corners free → rounded',
     lambda: mid_edge_tab_rounded()),

    # Rounded corners
    ('34_rounded_L',       'L-bracket 90°  r=10 mm — C0 & C3 free, C1 & C2 sharp',
     lambda: rounded_l_bracket()),
    ('35_rounded_one_arm', 'Panel + bottom arm 90°  r=8 mm — C2 & C3 free, C0 & C1 sharp',
     lambda: rounded_base_one_arm()),
    ('36_rounded_hex',     'Hexagonal panel  r=8 mm  1 edge extended, 4 free corners rounded',
     lambda: rounded_hex_panel()),
]


if __name__ == '__main__':
    print(f'Generating {len(SAMPLES)} samples → {OUT}/')
    for name, title, builder in SAMPLES:
        m = builder()
        save(m, name, title)
        m.print_tree()
        print()
    print('Done.')
