"""
isl_pairs.py

Likely intra-plane laser ISLs: each laser-capable satellite links to its
neighbours ahead and behind in the same orbital plane.

Usage:
    planes, plane_by_name, unassigned = find_planes(satrec_in_shell, when)
    shell.planes = planes                      # for every shell
    pairs = find_intra_isl_pairs(shells, satrec_by_name, version_by_name, when)
"""
import numpy as np

from simulation_param import MAX_LINK_RANGE_KM
from underlay_graph.util import to_julian

LASER_VERSIONS = {"v1.5", "v2 mini", "v3"}       # v1.0 has no lasers


def find_intra_isl_pairs(shells, satrec_by_name, version_by_name, when):
    """Return [(name_a, name_b), ...] for likely intra-plane laser ISLs, each pair once."""
    jd, fr = to_julian(when)
    pairs = []
    for shell in shells:
        for plane in shell.planes:
            lasers = [n for n in plane.sat_names if version_by_name.get(n) in LASER_VERSIONS]
            pairs += ring_links(lasers, satrec_by_name, jd, fr)
    return sorted(pairs)


def ring_links(names, satrec_by_name, jd, fr):
    """Order satellites along the plane and link each to the next, closing the ring."""
    pos = {}
    for n in names:
        err, r, v = satrec_by_name[n].sgp4(jd, fr)
        if not err:
            pos[n] = (np.array(r), np.array(v))
    ring = sorted(pos, key=lambda n: arg_of_latitude(*pos[n]))

    if len(ring) < 2:
        return []
    neighbours = list(zip(ring, ring[1:] + ring[:1]))     # the last one links back to the first
    if len(ring) == 2:
        neighbours = neighbours[:1]                       # don't list a-b twice
    return [tuple(sorted((a, b))) for a, b in neighbours
            if np.linalg.norm(pos[a][0] - pos[b][0]) <= MAX_LINK_RANGE_KM]


def arg_of_latitude(r, v):
    """Angle along the orbit from the ascending node, in degrees (0..360)."""
    h = np.cross(r, v)
    node = np.array([-h[1], h[0], 0.0])                   # points to the ascending node
    node /= np.linalg.norm(node)
    ahead = np.cross(h, node) / np.linalg.norm(h)         # 90° further along the orbit
    return np.degrees(np.arctan2(r @ ahead, r @ node)) % 360