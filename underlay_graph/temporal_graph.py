"""
temporal_graph.py

One NetworkX graph per simulation time step, built on demand.

    tg = build_temporal_graph(sc, aircraft_links, intra_isl_pairs,
                              sat_feeder_links, ground_links)
    G = tg.graph(i)            # graph at time step i (also: tg[i])

Nodes
    "sat:<name>"   satellites that appear in at least one link
    "gw:<name>"    gateways
    "pop:<name>"   PoPs
    "aircraft"     the aircraft

Node attributes
    type                      "satellite", "gateway", "pop" or "aircraft"
    lat_deg, lon_deg, alt_km  position
    lat_deg_per_s, lon_deg_per_s, alt_km_per_s
                              velocity (0 for gateways and PoPs)
    xyz_km                    ECEF position

Edge attributes
    type          "isl", "feeder", "aircraft" or "ground"
    distance_km   straight-line distance for space links,
                  distance along the Earth's surface for ground links

sc stores positions and altitudes in metres; everything here is in km.
"""
import numpy as np
import networkx as nx

from underlay_graph.feeder_links import geodetic_to_ecef_km
from underlay_graph.ground_links import pop_labels

M_TO_KM = 1e-3
EARTH_RADIUS_MEAN_KM = 6371.0
AIRCRAFT = "aircraft"


def build_temporal_graph(sc, aircraft_links, intra_isl_pairs, sat_feeder_links, ground_links):
    """Prepare everything needed to build one graph per time step."""
    return TemporalGraph(sc, aircraft_links, intra_isl_pairs, sat_feeder_links, ground_links)


class TemporalGraph:
    def __init__(self, sc, aircraft_links, isl_pairs, feeder_links, ground_links):
        self.unix_s = np.asarray(sc.unix_s, dtype=float)
        check_per_step(aircraft_links, len(self.unix_s), "aircraft_links")
        check_per_step(feeder_links, len(self.unix_s), "sat_feeder_links")
        self.aircraft_links = aircraft_links or [[] for _ in self.unix_s]
        self.feeder_links = feeder_links or [[] for _ in self.unix_s]
        self.isl_pairs = isl_pairs or []

        # Satellites: only those that appear in at least one link
        used = used_satellites(self.isl_pairs, self.feeder_links, self.aircraft_links)
        self.sat_names = [n for n in sc.sat_names if n in used]
        column = {name: k for k, name in enumerate(sc.sat_names)}
        self.sat_columns = np.array([column[n] for n in self.sat_names], dtype=int)

        # Satellite and aircraft trajectories (all time steps, as stored in sc)
        self.sat_lat, self.sat_lon, self.sat_alt_m = sc.sat_lla()
        self.sat_xyz_m = sc.sat_xyz
        self.aircraft_lla = np.asarray(sc.aircraft_lla, dtype=float)
        self.aircraft_xyz_m = np.asarray(sc.aircraft_xyz, dtype=float)

        # Ground nodes and links do not change over time
        self.gateways = ground_nodes(sc.gateways, list(sc.gateways["name"]))
        self.pops = ground_nodes(sc.pops, pop_labels(sc.pops))
        self.ground_edges = ground_edges(ground_links, self.gateways, self.pops)

    def __len__(self):
        return len(self.unix_s)

    def __getitem__(self, i):
        return self.graph(i)

    def graph(self, i):
        """NetworkX graph at time step i."""
        G = nx.Graph(time_index=i, unix_s=float(self.unix_s[i]))
        self.add_satellites(G, i)
        self.add_aircraft(G, i)
        self.add_ground_nodes(G)
        self.add_space_links(G, i)
        G.add_edges_from((u, v, {"type": "ground", "distance_km": d})
                         for u, v, d in self.ground_edges)
        return G

    # ---- nodes ---------------------------------------------------------------

    def add_satellites(self, G, i):
        cols = self.sat_columns
        lat, lon = self.sat_lat[i, cols], self.sat_lon[i, cols]
        alt_km = self.sat_alt_m[i, cols] * M_TO_KM
        lat_rate = self.rate(self.sat_lat[:, cols], i)
        lon_rate = self.rate(self.sat_lon[:, cols], i, angle=True)
        alt_rate = self.rate(self.sat_alt_m[:, cols], i) * M_TO_KM
        xyz_km = self.sat_xyz_m[i, cols] * M_TO_KM

        for k, name in enumerate(self.sat_names):
            G.add_node(f"sat:{name}", type="satellite",
                       lat_deg=float(lat[k]), lon_deg=float(lon[k]), alt_km=float(alt_km[k]),
                       lat_deg_per_s=float(lat_rate[k]), lon_deg_per_s=float(lon_rate[k]),
                       alt_km_per_s=float(alt_rate[k]), xyz_km=xyz_km[k])

    def add_aircraft(self, G, i):
        lat, lon, alt_m = self.aircraft_lla[i]
        lat_rate, lon_rate, alt_rate_m = (
            self.rate(self.aircraft_lla[:, 0], i),
            self.rate(self.aircraft_lla[:, 1], i, angle=True),
            self.rate(self.aircraft_lla[:, 2], i))
        G.add_node(AIRCRAFT, type="aircraft",
                   lat_deg=float(lat), lon_deg=float(lon), alt_km=float(alt_m * M_TO_KM),
                   lat_deg_per_s=float(lat_rate), lon_deg_per_s=float(lon_rate),
                   alt_km_per_s=float(alt_rate_m * M_TO_KM),
                   xyz_km=self.aircraft_xyz_m[i] * M_TO_KM)

    def add_ground_nodes(self, G):
        for kind, nodes in (("gateway", self.gateways), ("pop", self.pops)):
            prefix = "gw" if kind == "gateway" else "pop"
            for name, attrs in nodes.items():
                G.add_node(f"{prefix}:{name}", type=kind, **attrs,
                           lat_deg_per_s=0.0, lon_deg_per_s=0.0, alt_km_per_s=0.0)

    # ---- links ---------------------------------------------------------------

    def add_space_links(self, G, i):
        for a, b in self.isl_pairs:
            add_space_edge(G, f"sat:{a}", f"sat:{b}", "isl")
        for gw, sat in self.feeder_links[i]:
            add_space_edge(G, f"gw:{gw}", f"sat:{sat}", "feeder")
        for _, sat in self.aircraft_links[i]:
            add_space_edge(G, AIRCRAFT, f"sat:{sat}", "aircraft")

    # ---- helpers -------------------------------------------------------------

    def rate(self, series, i, angle=False):
        """Change per second at step i: central difference, one-sided at the ends.

        angle=True treats values as longitudes, so 179° -> -179° is +2°.
        """
        before, after = max(i - 1, 0), min(i + 1, len(self.unix_s) - 1)
        if before == after:
            return np.zeros_like(series[i], dtype=float)
        change = series[after] - series[before]
        if angle:
            change = (change + 180.0) % 360.0 - 180.0
        return change / (self.unix_s[after] - self.unix_s[before])


def add_space_edge(G, u, v, link_type):
    """Add a link with its straight-line length; skip it if a node is unknown."""
    if u in G and v in G:
        distance = np.linalg.norm(G.nodes[u]["xyz_km"] - G.nodes[v]["xyz_km"])
        G.add_edge(u, v, type=link_type, distance_km=float(distance))


def ground_nodes(df, names):
    """{name: {lat_deg, lon_deg, alt_km, xyz_km}} for gateways or PoPs."""
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    alt_km = df["alt_m"].to_numpy() * M_TO_KM if "alt_m" in df else np.zeros(len(df))
    xyz_km = geodetic_to_ecef_km(lat, lon, alt_km)
    return {name: dict(lat_deg=float(la), lon_deg=float(lo), alt_km=float(al), xyz_km=p)
            for name, la, lo, al, p in zip(names, lat, lon, alt_km, xyz_km)}


def ground_edges(ground_links, gateways, pops):
    """[(node_a, node_b, distance_km), ...] in the format of find_ground_links:
    a pair of two PoPs is a PoP-PoP link, otherwise the first name is a gateway."""
    edges = []
    for a, b in ground_links or []:
        if a in pops and b in pops:
            ends = (f"pop:{a}", pops[a]), (f"pop:{b}", pops[b])
        elif a in gateways and b in pops:
            ends = (f"gw:{a}", gateways[a]), (f"pop:{b}", pops[b])
        else:
            continue                                   # name not found
        (id_a, node_a), (id_b, node_b) = ends
        edges.append((id_a, id_b, surface_distance_km(node_a["xyz_km"], node_b["xyz_km"])))
    return edges


def surface_distance_km(p, q):
    """Great-circle distance between two points given in ECEF."""
    cos_angle = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)) * EARTH_RADIUS_MEAN_KM)


def used_satellites(isl_pairs, feeder_links, aircraft_links):
    """Names of all satellites that appear in at least one link."""
    used = {name for pair in isl_pairs for name in pair}
    for step in feeder_links:
        used.update(sat for _, sat in step)
    for step in aircraft_links:
        used.update(sat for _, sat in step)
    return used


def check_per_step(links, n_steps, name):
    if links is not None and len(links) != n_steps:
        raise ValueError(f"{name} has {len(links)} time steps, the scenario has {n_steps}")