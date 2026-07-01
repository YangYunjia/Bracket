# Sheet Metal Bracket Generator

A procedural generator for 3-D sheet metal bracket geometry, producing a
synthetic dataset of parametric mid-surface models.  Geometry is built with
**SheetMetalModel**; two rendering backends turn the same model into previews
or exact solids:

- **Numpy / Matplotlib** (`NumpyBracketModel`) — lightweight mesh tessellation
- **pythonOCC** (`OCC_Renderer`) — exact B-Rep solids with STEP / STL export

---

## Table of Contents

1. [Theory](#1-theory)
   - 1.1 [Mid-surface shell modelling](#11-mid-surface-shell-modelling)
   - 1.2 [Coordinate frames](#12-coordinate-frames)
   - 1.3 [Panel types](#13-panel-types)
   - 1.4 [Bend arc geometry](#14-bend-arc-geometry)
   - 1.5 [Curve3D and loft panels](#15-curve3d-and-loft-panels)
   - 1.6 [Recess geometry](#16-recess-geometry)
   - 1.7 [OCC B-Rep construction](#17-occ-b-rep-construction)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [API reference](#5-api-reference)
   - 5.1 [SheetMetalModel](#51-sheetmetalmodel)
   - 5.2 [Renderers](#52-renderers)
   - 5.3 [Primitives](#53-primitives)
6. [Sample generation](#6-sample-generation)

---

## 1. Theory

### 1.1 Mid-surface shell modelling

Sheet metal parts are thin enough that their geometry is well described by a
single **mid-surface** — a surface that sits midway through the material
thickness `t`.  Inner and outer faces are offset by `t/2` from the mid-surface.

This library models only the mid-surface.  Every part is assembled from:

| Element | Represents |
|---------|-----------|
| Panel | A flat or lofted region of the mid-surface |
| Bend arc | The curved strip connecting two panels along a shared edge |
| Recess | A drawn/stamped depression pressed into a panel |

The topology is a **forest of trees**: each root panel can have child panels
attached to any of its free edges via bend arcs; each child can in turn have
further children.  Connectivity is tracked in `PanelNode` trees.

---

### 1.2 Coordinate frames

Every flat panel defines a local right-handed orthonormal frame:

```
N = W × T          (outward normal of the panel)

        T  (v)
        ↑
        │   panel surface
   ─────┼──────────→ W  (u)
        origin
```

- **`origin`** — 3-D anchor point (corner 0).
- **`W`** — first in-plane axis; panel-local coordinate **u** runs along `W`.
- **`T`** — second in-plane axis; panel-local coordinate **v** runs along `T`.
- **`N = W × T`** — outward normal.

Panel-local coordinates `(u, v)` map to 3-D as:

```
P(u, v) = origin + u·W + v·T
```

**Edge numbering** (rectangular `FlatPanel`):

```
C3 ─── e2 (−W) ─── C2
│  e3 (−T)  e1 (+T) │
C0 ─── e0 (+W) ─── C1
```

Edge `i` connects vertex `i` to vertex `(i+1) mod 4` and returns
`(p0, p1, edge_dir, outward_dir)` via `panel.edge(i)`.

Polygon panels use the same CCW edge indexing: edge `i` runs from
`vertices[i]` to `vertices[(i+1) % n]`.

---

### 1.3 Panel types

#### FlatPanel

Rectangular panel parametrised by `length` (along `T`) and `width` (along `W`).
A special case of `PolygonPanel`.  Supports corner rounding (`corner_r`) at
**free** corners (corners whose adjacent edges are not occupied by a child).

#### PolygonPanel

Flat panel with an arbitrary CCW polygon outline given as `(N, 2)` vertices in
panel-local `(u, v)` coordinates.  Each edge is a `Curve3D.polyline` in world
space; the interior is triangulated with an ear-clip algorithm for the numpy
renderer.  Corner rounding uses the bisector formula at each free convex vertex.

#### LoftPanel

Panel defined by lofting between two or more **3-D profile curves**
(`Curve3D`).  The root profile attaches to a parent via a bend arc; the free
profile is where children extend from.  Optional intermediate profiles and
`t_start` / `t_end` tangent constraints steer G1 continuity at the profile
ends.

| `ruled` | Surface |
|---------|---------|
| `True`  | Ruled loft — straight lines between corresponding profile samples |
| `False` | Smooth skinned loft via `BRepOffsetAPI_ThruSections` |

Only profile edges **0** (root) and **2** (free) are extendable.

#### PipePanel

(Experimental) Like `LoftPanel` but swept along a spline spine with controlled
start/end tangents.  Rendered via `BRepOffsetAPI_MakePipeShell` in the OCC
backend, with a ThruSections fallback.

---

### 1.4 Bend arc geometry

All bends are represented by **`PartialBendArc`**.  A full-edge bend is the
degenerate case `BendArc` (`blend_r = 0`).

A bend arc connects the **parent edge** at angle `a = 0` to the **child root
edge** at angle `a = α`.  Every point on the bend surface sweeps along a
circular arc of radius `r_n` (the neutral-axis / bend radius):

```
P(s, a) = p_edge(s) + r_n · ((1 − cos a)·N_bend + sin a·T_in)
```

where:

- **`p_edge(s)`** — point at arc-length fraction `s` along the parent edge.
- **`T_in`** — outward tangent of the parent edge at the bend start.
- **`N_bend`** — signed panel normal (`±N`) selected by `bend_dir`.
- `a = 0` recovers the parent edge; `a = α` recovers the child root edge.

**Partial spans** use asymmetric blend radii `(blend_r_left, blend_r_right)`.
At the parent edge the arc widens by `blend_r` on each side, then tapers back
to the child width at `a = α` via:

```
ext(a, blend) = blend · f(a/α)
```

| `blend_mode` | Profile `f(t)` |
|--------------|----------------|
| `'arc'` (default) | `1 − √(t·(2−t))` |
| `'linear'` | `1 − t` |

When the blend radius is large relative to `α`, an effective radius
`eff_r_n = max(r_n, max(blend_r) / α)` ensures the arc length is sufficient.
Child panel root endpoints are `p0_out` / `p1_out` computed at `a = α`.

**`bend_dir`** selects which side of the panel the child folds toward:
`+1` toward `+N`, `−1` toward `−N`.

---

### 1.5 Curve3D and loft panels

#### Curve3D

Lightweight 3-D world-space curve used for panel edges and loft profiles:

| Constructor | Description |
|-------------|-------------|
| `Curve3D.line(p0, p1)` | Straight segment |
| `Curve3D.polyline(pts)` | Piecewise-linear through 3-D points |
| `Curve3D.arc(center, axis, radius, p_start, p_end)` | Circular arc |
| `Curve3D.bspline(control_pts, degree=3)` | Cubic (or general-degree) B-spline |

Key methods: `sample(n)`, `point_at_frac(f)`, `tangent_at_frac(f)`,
`arc_length()`, `insert_fracs(fs)`, `endpoints()`, `reversed()`.

The `_kind` attribute tells the OCC builder whether to emit an exact edge
(line / arc) or a dense polyline approximation (polyline / bspline).

#### Extending with a loft child

`SheetMetalModel.extend(..., panel_type='loft')` builds a `LoftPanel` (or
`'pipe'` for `PipePanel`) whose root profile is the arc's output edge
(`arc.p0_out → arc.p1_out`) and whose free profile is supplied as
`free_curve` (a `Curve3D` or list of 3-D points).  Optional
`intermediate_curves` add cross-sections between root and free.

Chain-bracket samples (`generate_chain_brackets.py`) demonstrate multi-profile
middle lofts with per-profile shift, twist, and width control.

---

### 1.6 Recess geometry

A **StampedRecess** is a drawn depression with this cross-sectional profile
(from panel surface inward):

```
─── panel surface ───────────────────────────────
    ╲ top bend (r_top, quarter-circle)
     │ straight wall (wall_h), optionally tilted by draft_angle
    ╱ bottom bend (r_bot, quarter-circle)
─── flat bottom ─────────────────────────────────
```

The footprint is a rounded rectangle with half-dimensions `(half_a, half_b)`
and corner radius `corner_r`.  At each depth level `k` along the profile, the
footprint is **inwardly offset** by the horizontal inset `u_k`:

```
half_a_k = half_a − u_k
half_b_k = half_b − u_k
corner_r_k = max(0, corner_r − u_k)
```

A **PolygonRecess** uses the same profile but with a general polygon footprint
instead of a rounded rectangle.  The inward offset is computed via the
bisector formula at each polygon vertex.

---

### 1.7 OCC B-Rep construction

The `OCC_Renderer` backend converts every mid-surface primitive to an
**exact OCC B-Rep shape** — no triangulated mesh wrapping.

#### Panels

| Panel type | OCC construction |
|------------|------------------|
| `FlatPanel` / `PolygonPanel` | Single planar face from `edge_curves` wires; rounded corners via `GC_MakeArcOfCircle` |
| `LoftPanel` | `BRepOffsetAPI_ThruSections`; aux-profile tangent steering when `t_start` / `t_end` set |
| `PipePanel` | `BRepOffsetAPI_MakePipeShell`; falls back to ThruSections on failure |

`Curve3D.line` → exact edge; `Curve3D.arc` → exact arc wire;
`Curve3D.bspline` / `polyline` → dense polygon wire approximation.

#### Bend arcs

`PartialBendArc` → ruled `BRepOffsetAPI_ThruSections` loft.  For each arc-angle
station `a`, a line-segment wire spans the (possibly blended) parent-edge
interval, producing an exact ruled surface that matches the numpy tessellation
formula.

#### Recesses

`BRepOffsetAPI_ThruSections` with `n_levels + 1` closed wires at successive
depth levels, plus a `BRepBuilderAPI_MakeFace` flat bottom face.

---

## 2. Repository layout

```
Bracket/
├── sheet_metal/                    # Main package
│   ├── __init__.py                 # Public re-exports
│   ├── basics/                     # Core geometry (NumPy + SciPy only)
│   │   ├── panel.py                # FlatPanel, PolygonPanel, LoftPanel, PipePanel
│   │   ├── bending.py              # PartialBendArc, BendArc
│   │   ├── curve.py                # Curve3D
│   │   ├── model.py                # SheetMetalModel
│   │   ├── recess.py               # StampedRecess, PolygonRecess
│   │   ├── tree.py                 # PanelNode connectivity tree
│   │   └── maths.py                # Vector / polygon helpers
│   └── rendering/                  # Visualization & export backends
│       ├── basics.py               # BaseRenderer (shared matplotlib export)
│       ├── numpy_model.py          # NumpyBracketModel
│       └── occ_model.py            # OCC_Renderer
│
├── sheet_metal.py                  # Legacy standalone demo script
│
├── generate_samples.py             # Bracket samples → output/
├── generate_occ_samples.py         # OCC versions → output_occ/ + STEP
├── generate_loft_samples.py        # LoftPanel twisted brackets → output_loft/
├── generate_chain_brackets.py      # Three-panel chain brackets → output_chain/
│
├── output/                         # (generated) NumPy backend PNG renders
├── output_occ/                     # (generated) OCC PNG renders
│   └── step/                       # (generated) STEP AP203 files
├── output_loft/                    # (generated) LoftPanel samples
│   └── step/
└── output_chain/                   # (generated) Chain-bracket samples
    └── step/
```

**Architecture.**  `SheetMetalModel` owns panels, arcs, recesses, and the
connectivity tree.  Renderers are stateless (aside from tessellation quality
settings) and operate on a built model:

```
model = SheetMetalModel(...)
# … build geometry …
groups, markers = renderer.render(model)
renderer.export_plot(groups, markers, ax=ax)   # matplotlib
renderer.export_step(groups, 'bracket.step')   # OCC only
```

---

## 3. Installation

### NumPy backend (lightweight preview)

Requires Python ≥ 3.9, NumPy, SciPy, and Matplotlib:

```bash
pip install numpy scipy matplotlib
```

### OCC backend (B-Rep + STEP export)

Requires the `occ312` conda environment (Python 3.12, OCC 7.9.0 / pythonOCC 7.7.2):

```bash
conda create -n occ312 python=3.12
conda activate occ312
conda install -c conda-forge pythonocc-core=7.7.2
pip install numpy scipy matplotlib
```

Verify:

```bash
conda run -n occ312 python -c "from OCC.Core.BRep import BRep_Builder; print('OCC OK')"
```

---

## 4. Quick start

### Build a model

```python
from sheet_metal import SheetMetalModel

model = SheetMetalModel(t=2.0, inner_r=5.0)

# Root panel: 100 × 80 mm flat plate
base = model.add_panel('base', [0, 0, 0], [1, 0, 0], [0, 1, 0],
                       length=100, width=80, corner_r=5.0)

# Fold a 40 mm flange from edge 1 at 90°
flange, arc = model.extend(base, edge_idx=1, angle_deg=90, child_length=40)

# Partial tab (middle 60 %) from edge 3 at −45°
tab, arc2 = model.extend(base, edge_idx=3, angle_deg=-45, child_length=30,
                         start_frac=0.2, end_frac=0.8)

# Stamped recess on the base panel
model.add_recess(base, cx=50, cy=40, width=30, height=20, corner_r=4,
                 r_top=3, r_bot=2, depth=8)

model.print_tree()
```

### NumPy / Matplotlib preview

```python
import matplotlib.pyplot as plt
from sheet_metal.rendering.numpy_model import NumpyBracketModel

renderer = NumpyBracketModel(n_arc_bend=28, n_blend=16)
groups, markers = renderer.render(model)

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
renderer.export_plot(groups, markers, ax=ax)
plt.show()
```

### OCC B-Rep + STEP export

```python
from sheet_metal.rendering.occ_model import OCC_Renderer

renderer = OCC_Renderer()
groups, markers = renderer.render(model)

renderer.export_step(groups, 'bracket.step')
renderer.export_stl(groups, 'bracket.stl')   # panels + arcs only

# Preview via OCC tessellation → matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
renderer.export_plot(groups, markers, ax=ax)
plt.show()
```

### Loft-panel child

```python
from sheet_metal import SheetMetalModel, Curve3D

model = SheetMetalModel(t=2.0, inner_r=5.0)
base  = model.add_panel('base', [0, 0, 0], [1, 0, 0], [0, 1, 0],
                        length=80, width=50)

# Arm with a 90° twist from horizontal root to vertical free edge
r_n   = 5.0
root  = ([0, 55, r_n], [80, 55, r_n])
free  = Curve3D.line([0, 55, r_n + 60], [80, 55, r_n + 60])

child, arc = model.extend(base, edge_idx=3, angle_deg=90,
                          panel_type='loft', free_curve=free, ruled=False)
```

See `generate_loft_samples.py` and `generate_chain_brackets.py` for richer
multi-profile loft examples.

---

## 5. API reference

### 5.1 SheetMetalModel

```python
model = SheetMetalModel(t=2.0, inner_r=5.0)
```

| Parameter | Meaning |
|-----------|---------|
| `t` | Material thickness (mm); used for neutral-axis radius in legacy contexts |
| `inner_r` | Default inner bend radius (mm) |

#### `add_panel(name, origin, T, W, length, width, corner_r=0.0) → FlatPanel`

Add a rectangular flat panel.  `name` is the first argument.

#### `add_polygon_panel(name, origin, T, W, vertices, corner_r=0.0) → PolygonPanel`

Add a flat panel with an arbitrary CCW polygon outline.  `vertices` is
`(N, 2)` in the `(u, v)` local frame.

#### `add_loft_panel(name, profiles, ruled=True) → LoftPanel`

Add a standalone loft panel (not connected to an existing panel).
`profiles` is a list of `Curve3D` objects or 3-D point lists.

#### `extend(parent, edge_idx, angle_deg, panel_type='flat', …) → (child, arc)`

Attach a child panel to edge `edge_idx` of `parent` via a bend arc.

| Parameter | Meaning |
|-----------|---------|
| `panel_type` | `'flat'` (default), `'loft'`, or `'pipe'` |
| `child_length` | Length of a flat child panel after the bend (mm) |
| `child_corner_r` | Corner rounding on the flat child |
| `free_curve` | `Curve3D` or 3-D points for the loft/pipe free profile |
| `intermediate_curves` | Optional list of intermediate loft profiles |
| `t_end` | Optional unit tangent at the free profile |
| `ruled` | `True` → ruled loft; `False` → smooth skin |
| `bend_dir` | `+1` fold toward `+N`, `−1` fold toward `−N` |
| `bend_r` | Bend radius (mm) |
| `start_frac`, `end_frac` | Fractional span along the edge (0–1) |
| `blend_r` | Lateral blend radius for partial bends (per-side; may be auto-reduced) |
| `blend_mode` | `'arc'` (default) or `'linear'` |
| `name` | Child panel name |

Returns `(child_panel, PartialBendArc)`.  Raises `ValueError` if the edge span
overlaps an existing child.

#### `extend_bend(…) → PartialBendArc`

Low-level bend-only call (no child panel).  Used by chain-bracket builders to
construct a middle loft from the arc endpoints.

#### `extend_loft_panel(name, arc, free_curve, …) → LoftPanel | PipePanel`

Build a loft/pipe child from an existing `PartialBendArc`.

#### `add_recess(panel, cx, cy, width, height, corner_r, r_top, r_bot, …) → StampedRecess`

Stamp a rounded-rectangle depression.  Specify either `depth` or `wall_h`.

#### `add_polygon_recess(panel, polygon, …) → PolygonRecess`

Same as `add_recess` but with a polygon footprint.

#### `mark_hole(panel, u, v, label='F', r_vis=5.0)`

Register a hole annotation at panel-local `(u, v)`.

#### `node_of(panel) → PanelNode` / `print_tree()`

Query or print the panel connectivity tree.

---

### 5.2 Renderers

Both renderers inherit `BaseRenderer.export_plot()` for matplotlib output.

#### NumpyBracketModel

```python
from sheet_metal.rendering.numpy_model import NumpyBracketModel

renderer = NumpyBracketModel(
    n_arc_bend=28,   # arc angle subdivisions
    n_blend=16,      # width subdivisions on partial bends
    n_sample=32,     # profile samples for LoftPanel
    n_boundary=80,   # recess boundary resolution
    n_profile=20,    # recess cross-section resolution
)
groups, markers = renderer.render(model)
renderer.export_plot(groups, markers, ax=ax)
```

#### OCC_Renderer

```python
from sheet_metal.rendering.occ_model import OCC_Renderer

renderer = OCC_Renderer(
    linear_deflection=0.5,
    angular_deflection=0.5,
    n_arc_bend=32,
)
groups, markers = renderer.render(model)
renderer.export_plot(groups, markers, ax=ax)       # tessellate → matplotlib
renderer.export_step(groups, 'bracket.step')       # STEP AP203
renderer.export_stl(groups, 'bracket.stl')         # binary STL (panels + arcs)
renderer.export_tessellated(groups, 'mesh.npz')    # NumPy archive of triangles
```

---

### 5.3 Primitives

#### Curve3D

See [§1.5](#15-curve3d-and-loft-panels).

#### FlatPanel / PolygonPanel

Common attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `origin` | `(3,)` | 3-D origin point |
| `T`, `W` | `(3,)` | In-plane unit vectors |
| `N` | `(3,)` | Outward unit normal (`W × T`) |
| `corner_r` | float | Free-corner rounding radius |
| `corner_trim` | list | Per-vertex trim lengths after filleting |

Key methods: `edge(i)`, `edge_curve(i) → Curve3D`, `edge_length(i)`,
`is_edge_occupied(...)`, `_register_child(...)`.

#### LoftPanel / PipePanel

Loft/pipe panels expose the same edge protocol as polygon panels for
`extend()`.  Only edges **0** (root profile) and **2** (free profile) on
`LoftPanel` are extendable.

#### PartialBendArc / BendArc

Key attributes: `p0`, `p1`, `T_in`, `T_out`, `W`, `N_bend`, `N_out`, `alpha`,
`r_n`, `eff_r_n`, `blend_r` (tuple), `blend_mode`, `p0_out`, `p1_out`.

`BendArc` is `PartialBendArc` with `blend_r = 0`.

#### StampedRecess / PolygonRecess

Built by `add_recess` / `add_polygon_recess`.  Expose `faces(...)` for the
numpy renderer.

---

## 6. Sample generation

### NumPy backend — bracket catalogue

```bash
python generate_samples.py
# Output: output/*.png  (three-view renders)
```

### OCC backend — same catalogue + STEP

```bash
conda run -n occ312 python generate_occ_samples.py
# Output: output_occ/*.png
#         output_occ/step/*.step
```

### LoftPanel twisted brackets

```bash
conda run -n occ312 python generate_loft_samples.py
# Output: output_loft/*.png, output_loft/step/*.step
```

Eight samples: twisted L- and U-brackets with uniform twist, taper, arc free
edges, and B-spline free edges.

### Chain brackets (flat → loft → flat)

```bash
conda run -n occ312 python generate_chain_brackets.py
# Output: output_chain/*.png, output_chain/step/*.step|.stl
```

Topology: `P1 (flat) ──[arc B1]── Middle (loft) ──[arc B2]── P3 (flat)`, with
optional partial side panels.  Edit the parameter block at the top of the
script (`T_THICK`, `MID_PROFILES`, bend radii, etc.) to change geometry.
Sample parameter sets are defined in `SAMPLES1` at the bottom of the file.

All sample scripts produce three-view renders (isometric / side / top) at
identical viewing angles for comparison.

### Sample catalogue (`generate_samples.py` / `generate_occ_samples.py`)

| Range | Category |
|-------|----------|
| 01–10 | Simple L, Z, U, T brackets; basic flanges |
| 11–20 | Multi-child edges; stepped flanges; C-sections |
| 21–30 | Stamped recesses; polygon recesses; hole annotations |
| 31–40 | Polygon panels; rounded outlines; multi-tab edges |
| 41–44 | Concave polygon panels |
| 45–50 | Curved-outline panels (legacy spline-edge builders) |

---

## Geometry notes

### Arc formula

The neutral-axis arc formula ensures every point on the bend surface lies at
radius `r_n` from the edge point in the `(N_bend, T_in)` plane:

```
P(s, a) = p_edge(s) + r_n · ((1 − cos a)·N_bend + sin a·T_in)
```

At `a = 0`: `P = p_edge(s)` (parent edge).
At `a = α`: `P = p_edge(s) + r_n · ((1 − cos α)·N + sin α·T_in)` (child root).

For partial bends, `eff_r_n` replaces `r_n` when the blend radius demands a
longer arc, and asymmetric `(blend_r_left, blend_r_right)` control the taper
on each side of the child span.

### Corner rounding

A rounded corner at vertex `v` with incoming direction `d_in` and outgoing
direction `d_out` is built with `GC_MakeArcOfCircle(P1, Pmid, P3)` where:

```
trim      = r · tan(θ/2),   θ = angle between d_in and d_out
arc_start = v − trim · d_in
arc_end   = v + trim · d_out
C         = arc_start + r · perp_CCW(d_in)   ← centre
arc_mid   = C + r · normalise(arc_start−C + arc_end−C)
```

`perp_CCW([x, y]) = [−y, x]` is the 90° CCW rotation in the panel plane.
Concave vertices and corners blocked by child panels remain sharp.
