
"""

This module contains the following classes:

Curves:
    Curve3D: 3-D world-space curve (line / arc / B-spline)
    # BSplineEdge    : cubic B-spline curve for one panel edge (cancelled now)

Panels:
    FlatPanel      : flat rectangular panel (as a special case of PolygonPanel)
    PolygonPanel   : flat panel with arbitrary polygon outline
    # SplinePanel  : flat panel with curved (B-spline) edge outlines (cancelled now)
    LoftPanel      : panel defined by lofting from parent to a child free edge (smoothed at the bending to enable C1 continuity)
    PipePanel      : (Experimental) panel defined by sweeping a spline spine with controlled start/end tangents

Bend arcs:
    PartialBendArc: partial-edge arc with smooth width taper (we eliminated the BendArc class to be implemented as a special case of PartialBendArc)
    # CurvedBendArc  : arc strip that sweeps along a curved (B-spline) edge (cancelled now)

Recesses:
    StampedRecess: recessed flat panel with a rounded top and bottom (not yet implemented)
    PolygonRecess: recessed polygon panel with a polygon top and bottom (not yet implemented)

"""