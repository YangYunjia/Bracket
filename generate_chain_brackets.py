"""
generate_chain_brackets.py — Three-panel chain bracket samples.

Run in the occ312 conda environment:
    conda run -n occ312 python generate_chain_brackets.py

Outputs:
  output_chain/<name>.png       — 3-view render (ISO / Side / Top)
  output_chain/step/<name>.step — STEP AP203 B-Rep file
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sheet_metal.rendering.occ_model import OCC_Renderer
from sheet_metal.template.chain import build_chain_bracket, prof


OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_chain')
STEP_OUT = os.path.join(OUT, 'step')
os.makedirs(OUT,      exist_ok=True)
os.makedirs(STEP_OUT, exist_ok=True)

VIEWS = [(28, -50, 'ISO'), (0, 90, 'Side'), (90, -90, 'Top')]
SAMPLES1 = [
    ('00_only_backbone',
     '',
     dict(p1_w=100, mid_profiles=[
         prof(20),
         prof(30),
         prof(50)],
        mid_ang=0.0, p3_ang=0.0,
        sp1_edge=0, sp1_ang=0.0, sp3_edge=2, sp3_ang=0.0)),
    ('00_twist',
     '',
     dict(p1_w=100, mid_profiles=[
         prof(20),
         prof(30, shift_n=15, twist=10),
         prof(50,twist=20)],
        mid_ang=20.0, p3_ang=-20.0,
        sp1_edge=0, sp1_ang=0.0, sp3_edge=2, sp3_ang=0.0)),
    ('00_bend',
     '',
     dict(p1_w=100, bend_r2=40.0, mid_profiles=[
         prof(20),
         prof(30, shift_n=15, twist=10),
         prof(50,twist=20)],
        mid_ang=70.0, p3_ang=-70.0,
        sp1_edge=0, sp1_ang=0.0, sp3_edge=2, sp3_ang=0.0)),
    ('side_flat',
     '',
     dict(p1_w=100, mid_profiles=[
         prof(20),
         prof(30),
         prof(50)],
        mid_ang=5.0, p3_ang=5.0,
        sp1_edge=1, sp1_ang=0.0,
        sp3_edge=1, sp3_ang=0.0)),
    ('side_twist',
     '',
     dict(p1_w=100, mid_profiles=[
         prof(20),
         prof(30),
         prof(50, twist2=10)],
        mid_ang=5.0, p3_ang=5.0,
        sp1_edge=1, sp1_ang=0.0,
        sp3_edge=1, sp3_ang=0.0)),
    ('chain_01_basic',
     'C-curve middle — 15 mm lateral bow at mid profile',
     dict(p1_w=100, mid_profiles=[
         prof(100),
         prof(150, shift_n=15, twist=10),
         prof(150, twist2=10)],
        mid_ang=30.0,
        sp1_edge=0, sp1_f0=0.05, sp1_f1=0.8,
        sp3_edge=1, sp3_f0=0.4, sp3_f1=0.6)),
]

if __name__ == '__main__':
    sample_set = SAMPLES1
    print(f'Generating {len(sample_set)} chain bracket samples…')
    renderer = OCC_Renderer(
        linear_deflection=0.1,
        angular_deflection=0.2,
        n_arc_bend=32,
    )
    tessellateds = {}

    for name, title, params in sample_set:
        m, fastener_panels = build_chain_bracket(params)
        groups, markers = renderer.render(m)
        tessellated = renderer.export_tessellated(groups)
        tessellateds[name] = {'facets': tessellated, 'fastener_panels': fastener_panels}

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
