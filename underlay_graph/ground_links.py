"""
ground_links.py

Static terrestrial links:
  * PoP backbone: a minimum spanning tree over all PoPs (so every PoP is
    reachable), plus links from each PoP to its nearest neighbours for
    redundancy. Not a full mesh.
  * Every gateway links to its geographically closest PoP.

Results use the same pair format as find_isl_pairs:
    [[pop, pop], ..., [gateway, pop], ...]

Usage:
    ground_links = find_ground_links(sc.gateways, sc.pops)
"""
import numpy as np
from simulation_param import EARTH_RADIUS_KM


def find_ground_links(gateways, pops, pop_neighbours: int = 2):
    """Return [[pop, pop], ..., [gateway, pop], ...] (static over the simulation).

    gateways       -- DataFrame with columns 'name', 'lat', 'lon' (degrees)
    pops           -- DataFrame with 'lat', 'lon' and 'name' (or 'city')
    pop_neighbours -- extra links from each PoP to its k nearest PoPs;
                      0 gives just the minimum spanning tree
    """
    pop_names = pop_labels(pops)
    pop_xyz = unit_vectors(pops["lat"], pops["lon"])
    gw_xyz = unit_vectors(gateways["lat"], gateways["lon"])

    # PoP backbone: spanning tree + nearest neighbours, each pair once
    pop_dist = great_circle_km(pop_xyz, pop_xyz)
    edges = minimum_spanning_tree(pop_dist) + nearest_neighbours(pop_dist, pop_neighbours)
    pop_links = sorted({tuple(sorted((pop_names[a], pop_names[b]))) for a, b in edges})

    # Every gateway to its closest PoP
    closest = great_circle_km(gw_xyz, pop_xyz).argmin(axis=1)
    gw_links = sorted([gw, pop_names[k]] for gw, k in zip(gateways["name"], closest))

    return [list(link) for link in pop_links] + gw_links


def pop_labels(pops):
    """PoP names: the 'name' column if there is one, else 'city'."""
    return list(pops["code"])


def unit_vectors(lat_deg, lon_deg):
    """Points on the unit sphere, (n, 3)."""
    lat, lon = np.radians(np.asarray(lat_deg)), np.radians(np.asarray(lon_deg))
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def great_circle_km(a, b):
    """Great-circle distances between every point in a (rows) and b (columns)."""
    return np.arccos(np.clip(a @ b.T, -1.0, 1.0)) * EARTH_RADIUS_KM


def minimum_spanning_tree(dist):
    """Edges (i, j) of the minimum spanning tree of a distance matrix (Prim)."""
    n = len(dist)
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()                 # shortest distance to the tree so far
    parent = np.zeros(n, dtype=int)       # tree node giving that distance
    edges = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges.append((int(parent[j]), j))
        in_tree[j] = True
        closer = dist[j] < best
        best[closer] = dist[j][closer]
        parent[closer] = j
    return edges


def nearest_neighbours(dist, k):
    """Edges (i, j) from every node to its k nearest other nodes."""
    if k <= 0 or len(dist) < 2:
        return []
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    return [(i, int(j)) for i in range(len(d)) for j in np.argsort(d[i])[:k]]