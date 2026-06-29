"""
sheet_metal.tree — panel connectivity tree.

Each FlatPanel produced by SheetMetalModel.extend() is represented as a
PanelNode that records its parent, the edge it was grown from, and the arc
that connects them.  Multiple roots are supported (models with more than one
independently placed base panel).
"""


class PanelNode:
    """Node in the panel connectivity tree.

    Attributes
    ----------
    panel       : FlatPanel
    parent      : PanelNode | None   (None for root panels)
    edge_idx    : int | None         (edge on the parent that was extended)
    arc         : BendArc | PartialBendArc | None
    start_frac  : float              (fraction along edge where child starts)
    end_frac    : float              (fraction along edge where child ends)
    children    : list[PanelNode]
    """

    def __init__(self, panel, parent=None, edge_idx=None, arc=None,
                 start_frac=0.0, end_frac=1.0):
        self.panel      = panel
        self.parent     = parent
        self.edge_idx   = edge_idx
        self.arc        = arc
        self.start_frac = float(start_frac)
        self.end_frac   = float(end_frac)
        self.children   = []

    # ── queries ───────────────────────────────────────────────────────────────

    def is_root(self):
        return self.parent is None

    def depth(self):
        d, n = 0, self
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    def ancestors(self):
        """Yield ancestor nodes from parent up to the root."""
        n = self.parent
        while n is not None:
            yield n
            n = n.parent

    # ── traversal ─────────────────────────────────────────────────────────────

    def iter_depth_first(self):
        """Yield self and all descendants in depth-first pre-order."""
        yield self
        for child in self.children:
            yield from child.iter_depth_first()

    def iter_breadth_first(self):
        """Yield self and all descendants in breadth-first order."""
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    # ── display ───────────────────────────────────────────────────────────────

    def _label(self):
        parts = [self.panel.name]
        if self.edge_idx is not None:
            parts.append(f'← e{self.edge_idx}')
        if self.start_frac > 1e-6 or self.end_frac < 1 - 1e-6:
            parts.append(f'[{self.start_frac:.0%}–{self.end_frac:.0%}]')
        return '  '.join(parts)

    def format_tree(self, _prefix='', _last=True):
        connector = ('└─ ' if _last else '├─ ') if self.parent else ''
        lines = [_prefix + connector + self._label()]
        child_prefix = _prefix + ('   ' if _last else '│  ')
        for i, child in enumerate(self.children):
            lines.append(child.format_tree(child_prefix, i == len(self.children) - 1))
        return '\n'.join(lines)

    def __repr__(self):
        return (f'PanelNode({self.panel.name!r}, '
                f'depth={self.depth()}, children={len(self.children)})')
