"""Unchanged v54-Q1 inscribed capability polygon, independent of dataset I/O."""
import numpy as np

def capability_facets(sides=16):
    """Return (A, b) for an inscribed unit disk: A @ [R_av,Q] <= b.

    Scale b by S_inv * mu_full * x for a task, or S_station for a station.
    sides divisible by four ensures symmetry in both axes and the downward
    active-power property required by 0 <= P_dispatch <= R_av.
    """
    if not isinstance(sides, int) or sides < 4 or sides % 4:
        raise ValueError("polygon_sides must be a positive multiple of four >= 4")
    theta = (2*np.arange(sides)+1)*np.pi/sides
    return np.column_stack((np.cos(theta), np.sin(theta))), np.full(sides, np.cos(np.pi/sides))

def capability_inequalities(rating_kVA, availability=1.0, sides=16):
    """Return A,rhs. For a variable x, multiply rhs by x in the MILP.

    availability is mu_full for the physical task model. M6 instead supplies
    its frozen pooled availability coefficient and selected-cohort sum.
    """
    if rating_kVA < 0 or not 0 <= availability <= 1:
        raise ValueError("nonnegative rating and availability in [0,1] required")
    A,b=capability_facets(sides)
    return A,b*rating_kVA*availability
