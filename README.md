# Sheet Metal Bracket Generator

A procedural generator for 3-D sheet metal bracket geometry, producing a
synthetic dataset of parametric mid-surface models with two rendering backends:

- **NumPy / Matplotlib** — lightweight mesh tessellation for fast preview
- **pythonOCC** — exact B-Rep (Boundary Representation) solids with STEP export

---

## Table of Contents

1. [Theory](#1-theory)
   - 1.1 [Mid-surface shell modelling](#11-mid-surface-shell-modelling)
   - 1.2 [Coordinate frames](#12-coordinate-frames)
   - 1.3 [Panel types](#13-panel-types)
   - 1.4 [Bend arc geometry](#14-bend-arc-geometry)
   - 1.5 [B-spline edges and curved bends](#15-b-spline-edges-and-curved-bends)
   - 1.6 [Recess geometry](#16-recess-geometry)
   - 1.7 [OCC B-Rep construction](#17-occ-b-rep-construction)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [API reference](#5-api-reference)
   - 5.1 [SheetMetalModel](#51-sheetmetalmodel)
   - 5.2 [OCC_BracketModel](#52-occ_bracketmodel)
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
| Panel | A flat (planar) region of the mid-surface |
| Bend arc | The curved strip connecting two panels along a shared edge |
| Recess | A drawn/stamped depression pressed into a panel |

The topology is a **forest of trees**: each root panel can have child panels
attached to any of its free edges via bend arcs; each child can in turn have
further children.

---

### 1.2 Coordinate frames

Every panel defines a local right-handed orthonormal frame:

```
N = T × W          (outward normal of the panel)

        W
        ↑
        │   panel surface
   ─────┼──────────→ T
        origin
```

- **`origin`** — a 3-D point at the panel's coordinate origin (corner C0).
- **`T`** — first in-plane axis (along edge 2, C0 → C1).
- **`W`** — second in-plane axis (along edge 0, C0 → C3).
- **`N = T × W`** — outward normal.

Panel-local coordinates `(u, v)` map to 3-D as:

```
P(u, v) = origin + u·T + v·W
```

**Edge numbering** (rectangular panels):

```
C3 ──── e3 (v = W) ──── C2
│  e0 (u=0)  e1 (u=L)   │
C0 ──── e2 (v = 0) ──── C1
```

Edge `i` returns `(p0, p1, edge_dir, outward_dir)`:

| Edge | Direction | Outward |
|------|-----------|---------|
| 0 | C0 → C3 (+W) | −T |
| 1 | C1 → C2 (+W) | +T |
| 2 | C0 → C1 (+T) | −W |
| 3 | C3 → C2 (+T) | +W |

---

### 1.3 Panel types

#### FlatPanel
Rectangular panel parametrised by `length` (along T) and `width` (along W).
Supports corner rounding (`corner_r`) at free corners (i.e., corners not
touching a child panel).

#### PolygonPanel
Flat panel with an arbitrary CCW polygon outline given as `(N, 2)` vertices in
panel-local (u, v) coordinates.  The interior is triangulated with an ear-clip
algorithm.  Corner rounding uses the bisector formula at each free vertex.

#### SplinePanel
Like `PolygonPanel` but each edge may optionally carry a `BSplineEdge` curve.
The polygon vertices still define the corner positions; the B-spline controls
the shape of the edge between corners.  Supports the same corner rounding as
`PolygonPanel` on straight-edge corners.

---

### 1.4 Bend arc geometry

A bend arc connects the **parent edge** at angle `a = 0` to the **child root
edge** at angle `a = α`.  Every point on the bend surface sweeps along a
circular arc of radius `r_n` (the neutral-axis radius):

```
r_n = inner_r + t/2
```

For a straight parent edge from `p0` to `p1`:

```
C(s)     = (1−s)·p0 + s·p1  +  r_n · N_bend        (arc centre at fraction s)
P(s, a)  = C(s) + r_n · (−cos(a)·N_bend + sin(a)·T_out)
```

where:
- **`N_bend`** — the bend direction (signed panel normal, `±N`).
- **`T_out`** — outward tangent of the arc; equals the outward normal of the
  parent edge rotated 90° CCW in the panel plane.
- `a = 0` recovers the parent edge, `a = α` recovers the child root edge.

**`PartialBendArc`** handles a child panel that spans only part of the parent
edge (`start_frac` to `end_frac`).  A smooth taper widens the bend surface by
`blend_r` at the parent edge and narrows it back to the child width at the
child edge, preventing sharp cuts at the blend boundary.  Two taper profiles
are supported:

- `'arc'` (default) — quarter-circle taper: `ext(a) = blend_r · (1 − √(t·(2−t)))`
- `'linear'` — linear ramp: `ext(a) = blend_r · (1 − a/α)`

**`bend_dir`** selects which side of the panel the child folds toward:
`+1` toward `+N`, `−1` toward `−N`.

---

### 1.5 B-spline edges and curved bends

#### BSplineEdge

A cubic (or user-specified degree) B-spline curve in the panel's local `(u, v)`
plane.  Internally it stores two `scipy.interpolate.BSpline` objects (one per
coordinate) with a **clamped uniform knot vector**:

```
knots = [0]*d  +  interior_knots  +  [1]*d
```

Clamped knots guarantee that `eval(0) = control_pts[0]` and
`eval(1) = control_pts[-1]`, so the curve endpoints coincide with the polygon
corners it is attached to.

Key methods:

| Method | Returns |
|--------|---------|
| `eval(t)` | `(u, v)` at parameter `t ∈ [0, 1]` |
| `tangent(t)` | Un-normalised `(du/dt, dv/dt)` |
| `arc_length(n)` | Approximate arc length by dense sampling |
| `sample_arclength(n)` | `(n, 2)` array of arc-length-uniform samples |
| `control_pts` | `(M, 2)` array of control points |
| `degree` | Polynomial degree (default 3) |

#### CurvedBendArc

When `extend()` is called from a `SplinePanel` edge that carries a
`BSplineEdge`, a `CurvedBendArc` is produced instead of a plain `BendArc`.

The surface is built by sweeping the arc formula along the curved edge.  At
each arc-length-uniform sample point `k`:

1. Evaluate the edge curve: `(u_k, v_k) = edge_curve.sample_arclength(n+1)[k]`
2. Lift to 3-D: `p_k = panel_origin + u_k·T + v_k·W`
3. Compute the local outward tangent from the 2-D curve tangent:
   `T_out_k = normalise(tang_v·T − tang_u·W)`
4. Sweep: `P(k, a) = (p_k + r_n·N_bend) + r_n·(−cos(a)·N_bend + sin(a)·T_out_k)`

This produces an `(n_arc+1) × (n_edge+1)` grid of 3-D points.

**Child panel**: `extend_panel` arc-transforms each control point of the parent
`BSplineEdge` to build the root edge of the child `SplinePanel` via
`_arc_transform_curve`.  The child panel is flat (planar) — an approximation
valid for small edge curvatures.

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

The profile `(u_k, v_k)` is sampled from `_sample_profile(r_top, wall_h,
r_bot, draft_angle, N)` where `v_k ≤ 0` is depth below the panel surface.

A **PolygonRecess** uses the same profile but with a general polygon footprint
instead of a rounded rectangle.  The inward offset is computed via the
bisector formula at each polygon vertex.

---

### 1.7 OCC B-Rep construction

The `sheet_metal_occ` backend converts every mid-surface primitive to an
**exact OCC B-Rep shape** — no triangulated mesh wrapping.

#### Panels

Each panel becomes a **single planar face** built with
`BRepBuilderAPI_MakeFace(wire, True)`:

- Straight edges → `BRepBuilderAPI_MakeEdge(P0, P1)`
- Rounded corners → `GC_MakeArcOfCircle(P1, Pmid, P3)` where the midpoint
  is `Pmid = C + r·normalise(v_start + v_end)` and the centre
  `C = arc_start + r·perp_CCW(d_in)`.
- B-spline edges → `Geom_BSplineCurve` constructed from OCC
  `TColgp_Array1OfPnt` (poles), distinct knot values, and their
  multiplicities (converted from the flat scipy knot vector).

#### BendArc — two selectable backends

Select via `arc_backend='loft'` (default) or `arc_backend='cyl'` on
`render()` / `export_step()` / `_build_occ_shapes()`.

**`'cyl'` — exact `Geom_CylindricalSurface`**

```
axis = gp_Ax3(C0, edge_dir, −N_bend)    C0 = p0 + r_n·N_bend
face = BRepBuilderAPI_MakeFace(cyl, −α, 0, 0, L, tol)
```

Exact and compact, but only valid for aligned, equal-length, parallel,
straight edges.

**`'loft'` — arc-guided ThruSections loft**

For each corresponding pair `(pA_j, cA_j)` on the parent and child edges:

```
C_j      = pA_j + r_n · N_bend
T_out_j  = (cA_j − C_j + r_n·cos α · N_bend) / (r_n · sin α)
P(a, j)  = C_j + r_n · (−cos a · N_bend + sin a · T_out_j)
```

At `a = 0` this recovers `pA_j` exactly; at `a = α` it recovers `cA_j`
exactly.  For an aligned straight `BendArc` all `T_out_j` are identical and
the loft surface is equivalent to the cylindrical surface.  For a
`CurvedBendArc`, `cA_j` is set to the **straight chord** of the arc endpoint
path (projected onto `W_child`), matching the flat child panel's root edge
and closing the gap between the bend surface and the child face.

`BRepOffsetAPI_ThruSections(isSolid=False, isRuled=False)` is called with
`n_arc + 1` polyline-wire profiles, producing a smooth lofted shell.

#### Recesses

`BRepOffsetAPI_ThruSections` is used with `n_levels + 1` closed wires at
successive depth levels, plus a `BRepBuilderAPI_MakeFace` flat bottom face.
STEP files produced by the OCC backend are ~8× smaller than equivalent
mesh-based STEP files.

---

## 2. Repository layout

```
Bracket/
├── sheet_metal/               # Core geometry (numpy + scipy only)
│   ├── __init__.py
│   ├── primitives.py          # FlatPanel, PolygonPanel, SplinePanel,
│   │                          # BendArc, PartialBendArc, CurvedBendArc,
│   │                          # BSplineEdge
│   ├── model.py               # SheetMetalModel, extend_panel,
│   │                          # _arc_transform_curve
│   └── recess.py              # StampedRecess, PolygonRecess
│
├── sheet_metal_occ/           # pythonOCC B-Rep backend
│   ├── __init__.py
│   └── occ_model.py           # OCC_BracketModel, all _*_occ builders,
│                              # tessellate_shapes
│
├── generate_samples.py        # Numpy backend: 50 samples → output/
├── generate_occ_samples.py    # OCC backend: 50 samples → output_occ/ + STEP
│
├── output/                    # (generated) PNG renders, numpy backend
└── output_occ/                # (generated) PNG renders, OCC backend
    └── step/                  # (generated) STEP AP203 files
```

---

## 3. Installation

### Numpy backend (lightweight preview)

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

### Numpy backend

```python
from sheet_metal import SheetMetalModel, BSplineEdge

m = SheetMetalModel(t=2.0, inner_r=5.0)

# Root panel: 100 × 80 mm flat plate
base = m.add_panel(origin=[0,0,0], T=[1,0,0], W=[0,1,0],
                   length=100, width=80, name='base', corner_r=5.0)

# Fold a 40 mm flange from edge 2 (bottom edge) at 90°
flange, arc = m.extend(base, edge_idx=2, angle_deg=90, child_length=40)

# Fold a partial tab (middle 60%) from edge 0 (left edge) at −45°
tab, arc2 = m.extend(base, edge_idx=0, angle_deg=-45, child_length=30,
                     start_frac=0.2, end_frac=0.8)

# Add a stamped recess to the base panel
m.add_recess(base, cx=50, cy=40, width=30, height=20, corner_r=4,
             r_top=3, r_bot=2, depth=8)

# Render (returns a matplotlib Axes3D)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
m.render(ax)
plt.show()
```

### OCC backend

```python
from sheet_metal_occ import OCC_BracketModel as SheetMetalModel
from sheet_metal import BSplineEdge

m = SheetMetalModel(t=2.0, inner_r=5.0)

base = m.add_panel(origin=[0,0,0], T=[1,0,0], W=[0,1,0],
                   length=100, width=80, name='base', corner_r=5.0)
m.extend(base, 2, 90, 40)

# Render with OCC tessellation → matplotlib
m.render()

# Export to STEP AP203
m.export_step('bracket.step')
```

`OCC_BracketModel` has the **same API** as `SheetMetalModel`; it is a
drop-in replacement.

### Curved-edge panel

```python
from sheet_metal_occ import OCC_BracketModel as SheetMetalModel
from sheet_metal import BSplineEdge

m = SheetMetalModel(t=2.0, inner_r=5.0)

# Convex B-spline bottom edge
crv = BSplineEdge([[0,0], [25,-20], [75,-20], [100,0]])

base = m.add_spline_panel(
    origin=[0,0,0], T=[1,0,0], W=[0,1,0],
    vertices=[[0,0],[100,0],[100,60],[0,60]],
    edge_curves=[crv, None, None, None],   # edge 0 is curved
    name='base',
)

# extend() automatically produces a CurvedBendArc
m.extend(base, edge_idx=0, angle_deg=90, child_length=45)

m.render()
m.export_step('curved_bracket.step')
```

---

## 5. API reference

### 5.1 SheetMetalModel

```python
m = SheetMetalModel(t=2.0, inner_r=5.0)
```

| Parameter | Meaning |
|-----------|---------|
| `t` | Material thickness (mm) |
| `inner_r` | Inner bend radius (mm); neutral-axis radius = `inner_r + t/2` |

#### `add_panel(origin, T, W, length, width, name=None, corner_r=0.0) → FlatPanel`

Add a rectangular flat panel.

#### `add_polygon_panel(origin, T, W, vertices, name=None, corner_r=0.0) → PolygonPanel`

Add a flat panel with an arbitrary polygon outline.  `vertices` is `(N, 2)` in
the T-W local frame; the polygon must be CCW.

#### `add_spline_panel(origin, T, W, vertices, edge_curves=None, name=None, corner_r=0.0) → SplinePanel`

Add a flat panel whose edges can carry B-spline curves.  `edge_curves` is a
list of `N` items (`BSplineEdge | None`); `None` means a straight edge.

#### `extend(parent, edge_idx, angle_deg, child_length, bend_dir=+1, start_frac=0.0, end_frac=1.0, blend_r=None, taper='arc') → (child, arc)`

Attach a child panel to edge `edge_idx` of `parent` via a bend arc.

| Parameter | Meaning |
|-----------|---------|
| `edge_idx` | Edge index (0–3 for rectangular panels) |
| `angle_deg` | Bend angle in degrees (positive or negative) |
| `child_length` | Length of the child panel after the bend (mm) |
| `bend_dir` | `+1` fold toward `+N`, `−1` fold toward `−N` |
| `start_frac` | Fractional start along the edge (0–1); default full edge |
| `end_frac` | Fractional end along the edge (0–1); default full edge |
| `blend_r` | Lateral blend half-width for partial bends (default = `r_n`) |
| `taper` | Blend profile: `'arc'` (default) or `'linear'` |

Returns `(child_panel, arc)`.  When the parent edge carries a `BSplineEdge`,
returns `(SplinePanel, CurvedBendArc)`; otherwise `(FlatPanel, BendArc)` or
`(FlatPanel, PartialBendArc)` for partial spans.

#### `add_recess(panel, cx, cy, width, height, corner_r, r_top, r_bot, depth=None, wall_h=None, draft_angle=0.0) → StampedRecess`

Stamp a rounded-rectangle depression into `panel`.  Specify either `depth`
(total depth including bends) or `wall_h` (straight wall height).

#### `add_polygon_recess(panel, polygon, corner_r=0.0, r_top, r_bot, depth=None, wall_h=None, draft_angle=0.0) → PolygonRecess`

Same as `add_recess` but with a polygon footprint.

#### `mark_hole(panel, u, v, label='F', r_vis=5.0)`

Register a hole annotation (centre + normal marker) at panel-local `(u, v)`.
Rendered as a coloured circle + arrow in visualisations.

#### `render(ax=None, ...) → Axes3D`

Render the full model into a matplotlib 3-D axes.  Returns the axes.

#### `print_tree()`

Print the panel connectivity tree (parent → children) to stdout.

---

### 5.2 OCC_BracketModel

Drop-in replacement for `SheetMetalModel` with the same full API, plus:

#### `render(..., arc_backend='loft') → Axes3D`

| `arc_backend` | Behaviour |
|---------------|-----------|
| `'loft'` (default) | Arc-guided `ThruSections` loft; handles non-aligned, curved, or partial-span edges |
| `'cyl'` | Exact `Geom_CylindricalSurface`; valid only for aligned equal-length straight `BendArc` |

#### `export_step(filepath, arc_backend='loft') → bool`

Write all shapes to a STEP AP203 file.  Returns `True` on success.

#### `_build_occ_shapes(arc_backend='loft') → (panel_shapes, arc_shapes, recess_shapes)`

Low-level access to the list of OCC `TopoDS_Shape` objects.

---

### 5.3 Primitives

#### BSplineEdge

```python
crv = BSplineEdge(control_pts, degree=3)
```

`control_pts` is `(M, 2)` in the panel's local `(u, v)` frame.
`M ≥ degree + 1`.  Both endpoints are interpolated exactly (clamped knots).

| Method | Description |
|--------|-------------|
| `eval(t)` | `(u, v)` at `t ∈ [0, 1]` |
| `tangent(t)` | Un-normalised `(du/dt, dv/dt)` |
| `arc_length(n=256)` | Approximate arc length |
| `sample(n)` | `(n, 2)` uniform-t samples |
| `sample_arclength(n)` | `(n, 2)` arc-length-uniform samples |
| `control_pts` | `(M, 2)` control point array |
| `degree` | Polynomial degree |

#### FlatPanel / PolygonPanel / SplinePanel

Common attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `origin` | `(3,)` | 3-D origin point |
| `T` | `(3,)` | First in-plane unit vector |
| `W` | `(3,)` | Second in-plane unit vector |
| `N` | `(3,)` | Outward unit normal (`T × W`) |
| `t` | float | Material thickness |
| `corner_r` | float | Free-corner rounding radius |

Key methods: `edge(i)`, `faces()`, `_corner_blocked(i)`, `_register_child(...)`.

`SplinePanel` additionally exposes `edge_curve(i) → BSplineEdge | None`.

#### BendArc / PartialBendArc / CurvedBendArc

Common attributes: `N_bend`, `alpha`, `r_n`.  All expose `faces()` returning
a list of quads/triangles for the numpy renderer.

---

## 6. Sample generation

### Numpy backend — 50 samples

```bash
python generate_samples.py
# Output: output/01_simple_L.png … output/50_spline_rounded.png
```

### OCC backend — 50 samples + STEP

```bash
conda run -n occ312 python generate_occ_samples.py
# Output: output_occ/*.png  (three-view renders via OCC tessellation)
#         output_occ/step/*.step  (STEP AP203 B-Rep files)
```

Both scripts produce three-view renders (isometric / side elevation / top
plan) at 3-D identical viewing angles for comparison.

### Sample catalogue

| Range | Category |
|-------|----------|
| 01–10 | Simple L, Z, U, T brackets; basic flanges |
| 11–20 | Multi-child edges; stepped flanges; C-sections |
| 21–30 | Stamped recesses; polygon recesses; hole annotations |
| 31–40 | Polygon panels (hexagon, L-shape, arm); rounded outlines |
| 41–44 | Concave polygon panels |
| 45–50 | SplinePanel / CurvedBendArc (B-spline curved edges) |

### Programmatic access

```python
# Iterate over all 50 sample builders
import generate_occ_samples as g
for name, title, builder in g.SAMPLES:
    model = builder()          # OCC_BracketModel
    model.export_step(f'{name}.step')
```

`SAMPLES` is a list of `(name, description, callable)` tuples defined at the
bottom of `generate_occ_samples.py`.

---

## Geometry notes

### Arc formula derivation

The neutral-axis arc formula ensures that every point on the bend surface lies
at constant radius `r_n` from the bend axis.  At fraction `s` along the edge:

```
C(s)    = edge_point(s) + r_n · N_bend       ← arc centre
P(s, a) = C(s) + r_n · (−cos a · N_bend + sin a · T_out)
```

At `a = 0`: `P = C − r_n·N_bend = edge_point(s)` (parent edge).
At `a = α`: `P = C + r_n·(−cos α · N_bend + sin α · T_out)` (child edge).

### OCC loft formula derivation

Given arbitrary parent/child edge samples `(pA_j, cA_j)`, the local T_out is
back-derived so that the arc passes through both endpoints exactly:

```
T_out_j = (cA_j − C_j + r_n · cos α · N_bend) / (r_n · sin α)
```

Substituting back: at `a = α`,
`P = C_j + r_n·(−cos α·N + sin α·T_out_j) = cA_j`. ✓

For an aligned straight `BendArc` all `T_out_j` are equal, recovering the
cylindrical formula.  For a `CurvedBendArc`, `cA_j` is the straight chord of
the arc endpoint path (projected onto `W_child = chord_dir`), ensuring the
loft terminates flush with the flat child panel's root edge.

### Corner rounding

A rounded corner at vertex `v` with incoming direction `d_in` and outgoing
direction `d_out` is built with `GC_MakeArcOfCircle(P1, Pmid, P3)` where:

```
trim        = r · tan(θ/2),   θ = angle between d_in and d_out
arc_start   = v − trim · d_in
arc_end     = v + trim · d_out
C           = arc_start + r · perp_CCW(d_in)   ← centre
arc_mid     = C + r · normalise(arc_start−C + arc_end−C)
```

`perp_CCW([x, y]) = [−y, x]` is the 90° CCW rotation in the panel plane.
