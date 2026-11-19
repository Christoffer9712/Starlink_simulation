"""
isl_pairs.py

Likely intra-plane laser ISLs between Starlink satellites.

Each laser-capable satellite links to its neighbours ahead and behind in the
same orbital plane. Planes, and the order of satellites along each plane,
come from operational.find_planes. Links are bidirectional, so each pair
appears once.

Usage:
    shells = find_shells(...)
    find_planes(shells, satrec_by_name, when)
    pairs = find_isl_pairs(shells, satrec_by_name, version_by_name, when)
"""
from datetime import datetime
from typing import Optional

import numpy as np

from simulation_param import *
from underlay_graph.util import newest_epoch, to_julian

LASER_VERSIONS = {"v1.5", "v2 mini", "v3"}


def find_isl_pairs(shells, satrec_by_name, version_by_name,
                   when: Optional[datetime] = None):
    """Return [[name_a, name_b], ...] for likely intra-plane laser ISLs.

    shells          -- shells with planes filled in by find_planes
    satrec_by_name  -- {name: Satrec}, used for the link-range check
    version_by_name -- {name: version}; only laser-capable versions get links
    when            -- UTC datetime; use the same as for find_planes
    """
    names = [n for shell in shells for plane in shell.planes for n in plane.names]
    if not names:
        return []
    jd, fr = (to_julian(when) if when
              else newest_epoch(satrec_by_name[n] for n in names))

    pairs = []
    for shell in shells:
        for plane in shell.planes:
            if len(plane.names) < MIN_PLANE_SATELLITES:
                continue                  # satellites in transit between planes
            ring = [n for n in plane.names if has_laser_isl(version_by_name.get(n))]
            pairs += ring_links(ring, satrec_by_name, jd, fr)
    return sorted(pairs)


def has_laser_isl(version):
    return version in LASER_VERSIONS


def ring_links(ring, satrec_by_name, jd, fr):
    """Link consecutive satellites along the plane, closing the ring."""
    if len(ring) < 2:
        return []
    neighbours = list(zip(ring, ring[1:] + ring[:1]))
    if len(ring) == 2:
        neighbours = neighbours[:1]               # avoid listing a-b twice
    return [sorted([a, b]) for a, b in neighbours
            if link_range_km(satrec_by_name[a], satrec_by_name[b], jd, fr)
            <= MAX_LINK_RANGE_KM]


def link_range_km(satrec_a, satrec_b, jd, fr):
    """Distance between two satellites in km (inf if either can't be propagated).

    SGP4's TEME positions are used directly: distances are the same in ECEF.
    """
    error_a, r_a, _ = satrec_a.sgp4(jd, fr)
    error_b, r_b, _ = satrec_b.sgp4(jd, fr)
    if error_a or error_b:
        return float("inf")
    return float(np.linalg.norm(np.subtract(r_a, r_b)))