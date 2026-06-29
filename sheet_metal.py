"""
sheet_metal.py — 2-D panel-extension model for sheet metal brackets.

Uniform-thickness shell representation: only the mid-surface (neutral axis)
is generated.  Each panel is a single flat quad; each bend arc is a single
curved strip.  Thickness is only used to compute the neutral-axis bend radius
r_n = inner_r + t/2.

Panel edge numbering
────────────────────
    C3 ─── e3 (v=W_) ─── C2
    │  e0 (u=0)  e1 (u=L)  │
    C0 ─── e2 (v=0)  ─── C1

  e0  C0→C3   outward = -T
  e1  C1→C2   outward = +T
  e2  C0→C1   outward = -W
  e3  C3→C2   outward = +W

Usage
─────
  m = SheetMetalModel(t=2.0, inner_r=5.0)
  base   = m.add_panel([0,0,0], [1,0,0], [0,1,0], length=100, width=60)
  flange, _ = m.extend(base, edge_idx=1, angle_deg=90, child_length=60,
                       start_frac=0.2, end_frac=0.8)
  m.render()
  plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── helpers ───────────────────────────────────────────────────────────────────

def _n(v):
    v = np.asarray(v, float)
    m = np.linalg.norm(v)
    return v / m if m > 1e-12 else v


# ── FlatPanel ─────────────────────────────────────────────────────────────────

class FlatPanel:
    """Flat rectangular panel (mid-surface only)."""

    def __init__(self, origin, T, W, length, width, t=2.0, name='P'):
        self.origin = np.asarray(origin, float)
        self.T      = _n(T)
        self.W      = _n(W)
        self.length = float(length)
        self.width  = float(width)
        self.t      = float(t)
        self.name   = str(name)

    @property
    def N(self):
        return _n(np.cross(self.T, self.W))

    def corners(self):
        o, T, W, L, Wd = self.origin, self.T, self.W, self.length, self.width
        return [o, o+L*T, o+L*T+Wd*W, o+Wd*W]   # C0,C1,C2,C3

    def edge(self, i):
        """(p0, p1, edge_dir, outward_dir) for edge i."""
        C = self.corners()
        T, W = self.T, self.W
        data = [
            (C[0], C[3],  W, -T),
            (C[1], C[2],  W,  T),
            (C[0], C[1],  T, -W),
            (C[3], C[2],  T,  W),
        ]
        p0, p1, ed, out = data[i]
        return p0.copy(), p1.copy(), _n(ed), _n(out)

    def faces(self):
        """Single mid-surface quad."""
        C = self.corners()
        return [[C[0], C[1], C[2], C[3]]]


# ── BendArc ───────────────────────────────────────────────────────────────────

class BendArc:
    """Full-edge cylindrical arc (mid-surface strip)."""

    def __init__(self, p0, p1, T_out, N_bend, alpha, r_n, n_arc=28):
        self.p0     = np.asarray(p0, float)
        self.p1     = np.asarray(p1, float)
        self.T_out  = _n(T_out)
        self.N_bend = _n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        self.n_arc  = int(n_arc)

    def _pt(self, s, a):
        p_edge = (1-s)*self.p0 + s*self.p1
        C      = p_edge + self.r_n * self.N_bend
        return C + self.r_n * (-np.cos(a)*self.N_bend + np.sin(a)*self.T_out)

    def faces(self):
        alphas = np.linspace(0, self.alpha, self.n_arc+1)
        return [[self._pt(0,alphas[i]),  self._pt(1,alphas[i]),
                 self._pt(1,alphas[i+1]), self._pt(0,alphas[i+1])]
                for i in range(self.n_arc)]


# ── PartialBendArc ────────────────────────────────────────────────────────────

class PartialBendArc:
    """Partial-edge cylindrical arc with a smooth tapered transition.

    One continuous bending surface — no separate pieces.

    The child panel connects along [s_start, s_end] (child_width = s_end − s_start).
    When child_width < parent edge length the arc widens at the base (a = 0) by
    blend_r on each side, giving an effective connection length of
    child_width + 2·blend_r at the parent edge.  The width tapers back to
    child_width at a = alpha via ext(a) = blend_r · f(a/alpha), where f is
    determined by taper:

        'arc'    (default) — normalized quarter-circle:  f(t) = √(1 − t²)
        'linear'           — straight ramp:               f(t) = 1 − t

    Both guarantee ext(0) = blend_r and ext(alpha) = 0.

    The s-range of the surface at arc angle a is

        [s_start − ext(a),  s_end + ext(a)]   (clamped to the parent edge).

    The surface is tessellated as an (n_arc × n_blend) quad grid.
    """

    def __init__(self, p0_full, p1_full,
                 s_start, s_end,
                 T_out, N_bend,
                 alpha, r_n,
                 blend_r=None, taper='arc', n_arc=32, n_blend=16):
        self.p0     = np.asarray(p0_full, float)
        self.p1     = np.asarray(p1_full, float)
        self.L      = float(np.linalg.norm(self.p1 - self.p0))
        self.s0     = float(s_start)
        self.s1     = float(s_end)
        self.T_out  = _n(T_out)
        self.N_bend = _n(N_bend)
        self.alpha  = float(alpha)
        self.r_n    = float(r_n)
        self.blend  = float(blend_r) if blend_r is not None else float(r_n)
        self.taper  = taper
        self.n_arc  = int(n_arc)
        self.n_bld  = int(n_blend)

    def _edge_pt(self, s):
        frac = np.clip(s / self.L, 0.0, 1.0)
        return (1 - frac)*self.p0 + frac*self.p1

    def _pt(self, s, a):
        C = self._edge_pt(s) + self.r_n * self.N_bend
        return C + self.r_n * (-np.cos(a)*self.N_bend + np.sin(a)*self.T_out)

    def _ext(self, a):
        t = a / self.alpha
        if self.taper == 'linear':
            scale = 1.0 - t
        else:
            scale = 1.0 - np.sqrt(max(0.0, t*(2-t)))
        return self.blend * scale

    def faces(self):
        na = self.n_arc
        ns = self.n_bld
        a_vals = np.linspace(0.0, self.alpha, na + 1)

        grid = []
        for a in a_vals:
            ext  = self._ext(a)
            s_lo = max(0.0,    self.s0 - ext)
            s_hi = min(self.L, self.s1 + ext)
            grid.append([self._pt(s, a)
                         for s in np.linspace(s_lo, s_hi, ns + 1)])

        return [[grid[i][j], grid[i][j+1], grid[i+1][j+1], grid[i+1][j]]
                for i in range(na) for j in range(ns)]


# ── extend_panel ──────────────────────────────────────────────────────────────

def extend_panel(parent, edge_idx, angle_deg, child_length,
                 inner_r=5.0, bend_dir=+1,
                 start_frac=0.0, end_frac=1.0,
                 blend_r=None):
    """
    Extend a new panel from edge `edge_idx` of `parent`.

    start_frac / end_frac  : portion of the edge to span  (0–1)
    blend_r                : smooth-taper length at each end of a partial
                             extension (default = r_n = inner_r + t/2)
    bend_dir               : +1 toward +N_parent, −1 toward −N_parent

    Returns (child : FlatPanel, arc : BendArc | PartialBendArc)
    """
    p0_full, p1_full, edge_dir, T_out = parent.edge(edge_idx)
    L_edge  = float(np.linalg.norm(p1_full - p0_full))

    p0 = p0_full + start_frac * (p1_full - p0_full)
    p1 = p0_full + end_frac   * (p1_full - p0_full)

    t     = parent.t
    alpha = np.radians(abs(angle_deg))
    bdir  = float(bend_dir) * parent.N
    r_n   = inner_r + t / 2
    print(parent.name, parent.N, bdir, T_out)

    T_child     = _n(np.sin(alpha)*bdir + np.cos(alpha)*T_out)
    W_child     = edge_dir
    child_width = float(np.linalg.norm(p1 - p0))

    # Child origin: neutral-axis arc endpoint at the p0 side
    C0           = p0 + r_n * bdir
    child_origin = C0 + r_n * (-np.cos(alpha)*bdir + np.sin(alpha)*T_out)

    child = FlatPanel(
        origin=child_origin, T=T_child, W=W_child,
        length=child_length, width=child_width,
        t=t, name=f'{parent.name}.e{edge_idx}',
    )

    is_partial = (start_frac > 1e-6) or (end_frac < 1.0 - 1e-6)

    if is_partial:
        _br = float(blend_r) if blend_r is not None else r_n
        arc = PartialBendArc(
            p0_full=p0_full, p1_full=p1_full,
            s_start=start_frac * L_edge,
            s_end=end_frac     * L_edge,
            T_out=T_out, N_bend=bdir,
            alpha=alpha, r_n=r_n,
            blend_r=_br,
        )
    else:
        arc = BendArc(p0=p0, p1=p1, T_out=T_out, N_bend=bdir,
                      alpha=alpha, r_n=r_n)

    return child, arc


# ── SheetMetalModel ───────────────────────────────────────────────────────────

class SheetMetalModel:

    def __init__(self, t=2.0, inner_r=5.0):
        self.t       = float(t)
        self.inner_r = float(inner_r)
        self.panels  = []
        self.arcs    = []
        self._holes  = []

    def add_panel(self, origin, T, W, length, width, name=None):
        if name is None:
            name = f'P{len(self.panels)}'
        p = FlatPanel(origin, T, W, length, width, self.t, name)
        self.panels.append(p)
        return p

    def extend(self, parent, edge_idx, angle_deg, child_length,
               bend_dir=+1, start_frac=0.0, end_frac=1.0, blend_r=None):
        child, arc = extend_panel(
            parent, edge_idx, angle_deg, child_length,
            inner_r=self.inner_r, bend_dir=bend_dir,
            start_frac=start_frac, end_frac=end_frac, blend_r=blend_r,
        )
        self.panels.append(child)
        self.arcs.append(arc)
        return child, arc

    def mark_hole(self, panel, u, v, label='F', r_vis=5.0):
        centre = panel.origin + u*panel.T + v*panel.W
        self._holes.append({'centre': centre, 'normal': panel.N,
                            'label': label, 'r': r_vis})

    def render(self, ax=None, color_panel='#4a90c4', color_bend='#2e6ea6',
               alpha_surf=0.85, ec='#1a3a52', lw=0.4, elev=28., azim=-50.):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection='3d')

        for p in self.panels:
            ax.add_collection3d(Poly3DCollection(
                p.faces(), alpha=alpha_surf,
                facecolor=color_panel, edgecolor=ec, linewidth=lw))

        for arc in self.arcs:
            ax.add_collection3d(Poly3DCollection(
                arc.faces(), alpha=alpha_surf,
                facecolor=color_bend, edgecolor=ec, linewidth=lw))

        hole_colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for i, h in enumerate(self._holes):
            c, ctr, n, r = hole_colors[i % 4], h['centre'], h['normal'], h['r']
            ref = np.array([0,1,0]) if abs(n[1]) < 0.85 else np.array([1,0,0])
            u_ = _n(np.cross(n, ref))
            v_ = _n(np.cross(n, u_))
            ang  = np.linspace(0, 2*np.pi, 48)
            ring = ctr + r*(np.outer(np.cos(ang), u_) + np.outer(np.sin(ang), v_))
            ax.plot(ring[:,0], ring[:,1], ring[:,2], color=c, lw=2.2, zorder=10)
            ax.quiver(*ctr, *(n*r*1.5), color=c, lw=1.8,
                      arrow_length_ratio=0.35, zorder=10)
            ax.text(*(ctr+n*r*2.5), h['label'],
                    fontsize=10, color=c, fontweight='bold')

        # Auto-scale
        pts = [pt for p   in self.panels for f in p.faces()   for pt in f] + \
              [pt for arc in self.arcs   for f in arc.faces() for pt in f]
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        mid    = (lo + hi) / 2
        span   = (hi - lo).max() * 0.65
        for fn, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
            fn(m - span, m + span)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        return ax


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default='output/partial_extension.png')
    args = ap.parse_args()

    m = SheetMetalModel(t=2.0, inner_r=5.0)
    base = m.add_panel([0,0,0], [1,0,0], [0,1,0],
                       length=100, width=60, name='base')

    # Partial flange: 20%–80% of the end edge, 90°, smooth tapered ends
    flange, _ = m.extend(base, edge_idx=1, angle_deg=90, child_length=60,
                         start_frac=0.2, end_frac=0.8)

    m.mark_hole(base,   50, 30,              'F1')
    m.mark_hole(flange, 30, flange.width/2,  'F2')

    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim, title) in enumerate([
        (25, -50, 'Isometric'),
        ( 0,  90, 'Side  (X→)'),
        (90, -90, 'Top'),
    ], 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        m.render(ax, elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        'Mid-surface shell · partial flange 20%–80% of end edge · '
        '90° bend · r_inner = 5 mm\n'
        'Continuous tapered arc  (connection length = child_width + 2·r_n, blend_r = r_n = 6 mm)',
        fontsize=10)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save)
    plt.savefig(out, dpi=170, bbox_inches='tight')
    print(f'Saved → {out}')
