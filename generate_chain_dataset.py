"""
generate_chain_dataset_from_00_bend.py

Generate a parameter dataset by sampling around the `00_bend` setting
from `generate_chain_brackets.py`.

What is sampled (uniformly):
  - p1_l, p1_w, bend_r2, mid_ang, p3_ang, p3_l, sp1_ang, sp3_ang
  - for each `prof` inside `mid_profiles`:
      shift_n, twist, twist2

All other parameters are taken from the template defaults in
`sheet_metal.template.chain` and/or the base `00_bend` config.

The script can optionally export:
  - composite PNG renders (3 views)
  - STEP / STL
  - tessellated facets (NPZ)

Run:
  python generate_chain_dataset_from_00_bend.py --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from sheet_metal.rendering.occ_model import OCC_Renderer
from sheet_metal.template.chain import _BASE as TEMPLATE_BASE
from sheet_metal.template.chain import build_chain_bracket, prof

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

N_PROFILE = 3
# ──────────────────────────────────────────────────────────────────────────────
# Sampling ranges (EDIT THESE)
# ──────────────────────────────────────────────────────────────────────────────
#
# Ranges are absolute values. Sampling is uniform over [low, high].

RANGES: dict[str, tuple[float, float]] = {
    "p1_l": (20.0, 100.0),
    "p1_w": (80.0, 140.0),
    "bend_r1": (20.0, 60.0),
    "bend_r2": (20.0, 60.0),
    "mid_ang": (-80.0, 80.0),
    "p3_ang": (-80.0, 80.0),
    "p3_l": (20.0, 100.0),
    "sp1_ang": (-40.0, 40.0),
    "sp3_ang": (-40.0, 40.0),
}
MID_RANGES: dict[str, tuple[float, float]] = {
    # Apply the same range to every `prof` in `mid_profiles`.
    "shift_t": (60, 100),
    "shift_n": (-50.0, 50.0),
    "shift_w": (-20.0, 20.0),
    "twist": (-20.0, 20.0),
    "twist2": (-20.0, 20.0),
}

VIEWS = [(28, -50, "ISO"), (0, 90, "Side"), (90, -90, "Top")]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_chain_dataset"),
        help="Output directory.",
    )
    parser.add_argument("--png", action="store_true", help="Export composite PNG renders.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png")
    tess_dir = os.path.join(args.out_dir, "tessellated.npz")

    if args.png:
        os.makedirs(png_dir, exist_ok=True)

    # Renderer is needed for geometry exports / rendering.
    renderer = OCC_Renderer(
        linear_deflection=0.1,
        angular_deflection=0.2,
        n_arc_bend=32,
    )

    rng_gen = np.random.default_rng(args.seed)
    n = int(args.n)
    raw_parameters = np.random.uniform(size=(n, len(RANGES) + N_PROFILE * len(MID_RANGES)))
    # convert parameters to the correct ranges
    parameters = {
        name: raw_parameters[:, idx] * (high - low) + low for idx, (name, (low, high)) in enumerate(RANGES.items())
    }
    for i in range(N_PROFILE):
        parameters.update({
            name + f"_{i}": raw_parameters[:, len(RANGES) + i * len(MID_RANGES) + idx] * (high - low) + low for idx, (name, (low, high)) in enumerate(MID_RANGES.items())
        })
    # save parameters to a parquet file
    parameters_df = pd.DataFrame(parameters)
    parameters_df.to_parquet(os.path.join(args.out_dir, "parameters.parquet"))

    tess_written = 0

    print(f"[ChainDataset] generating {n} samples → {args.out_dir}")
    tessellateds = []

    for i in range(n):
        name = f"ds_{i:06d}"
        params = parameters_df.iloc[i].to_dict()
        params["mid_profiles"] = [
            prof(**{
                name: params[f"{name}_{i}"] for name in MID_RANGES.keys()
            })
            for i in range(N_PROFILE)
        ]
        params.update(sp1_edge=0, sp3_edge=2)
        # Build geometry + render/export.
        m, fastener_panels = build_chain_bracket(params)
        groups, markers = renderer.render(m)

        tessellateds.append({'facets': renderer.export_tessellated(groups), 'fastener_panels': fastener_panels})
        tess_written += 1

        if (i + 1) % max(1, n // 100) == 0:
            print(f"  {i + 1}/{n} done")
            if args.png:
                fig = plt.figure(figsize=(14, 5))
                for k, (elev, azim, label) in enumerate(VIEWS, 1):
                    ax = fig.add_subplot(1, 3, k, projection="3d")
                    renderer.export_plot(groups, markers, ax=ax, elev=elev, azim=azim)
                    ax.set_title(label, fontsize=9)
                fig.suptitle(f"[ChainBracket] {name}", fontsize=10)
                plt.tight_layout()
                png_path = os.path.join(png_dir, f"{name}.png")
                plt.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
    
    np.savez(tess_dir, tessellateds=tessellateds)
    print(f"Done → {args.out_dir} (tessellated: {tess_written})")


if __name__ == "__main__":
    main()

