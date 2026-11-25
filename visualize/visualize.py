"""Animated globe of a Scenario: one animation frame per second of sim time.

Shown per frame
  * aircraft (moving) and its full recorded track
  * the satellites the aircraft is linked to in this time step, with a line
    to the aircraft (blue; from aircraft_links)
  * operating broadband satellites inside the view; with shells
    given, one legend group per shell (own colour) with its satellites,
    intra-plane ISLs, inter-plane ISLs and plane orbits (orbits hidden by
    default) - every item can be switched on and off in the legend
  * other operating satellites, e.g. direct-to-cell (amber)
  * non-operational satellites inside the view (red ×)
  * laser ISLs between two satellites that are both inside the view:
    intra-plane (purple) and inter-plane (light purple, dashed).
    ISLs between two different shells are drawn outside the shell groups.
  * gateway-satellite feeder links of this time step, when the satellite is
    inside the view (green, dotted)
  * the path used in the delay comparison (thick, pink-red) and, if different,
    the path to the nearest PoP (dashed); from compare(...)
Static: gateways, PoPs, the PoP backbone (dark), gateway backhaul to PoPs
(grey, dashed) and the plane orbits (drawn for the middle of the window).

A "Broadband only / All satellites" switch (top right) hides or shows the
other and non-operational satellites; their legend entries toggle them too.

Hovering over a satellite shows its name, version, inclination, shell and
plane, current altitude and elevation from the aircraft. Hovering over a path
point shows the node; the aircraft and the path start show the comparison
summary.

File size: only satellites inside the fixed view (plus a small margin) are
written to the frames, and coordinates are rounded to COORD_DECIMALS. Use
frame_step to thin long windows further.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from positions_sim.geometry import ecef_to_lla

COLORS = {
    "aircraft": "#eb6834",
    "sat_linked": "#2a78d6",
    "sat_horizon": "#a3a29c",
    "sat_other": "#d9a441",
    "sat_inactive": "#d64545",
    "isl_intra": "#8a4fc7",
    "isl_inter": "#b48ee0",
    "feeder": "#1baf7a",
    "backbone": "#2b2b29",
    "backhaul": "#6b6a64",
    "path": "#e0115f",
    "gateway": "#1baf7a",
    "pop": "#2b2b29",
    "track": "#eb6834",
}

# One colour per shell (satellites and orbits), repeated if there are more shells
SHELL_COLORS = ["#6b7b8c", "#17a2b8", "#a3a23a", "#8c564b",
                "#c2549d", "#5f6fd1", "#2e8b57", "#d2691e"]

HORIZON_DEG = 23.0         # Earth-central angle to a 550 km satellite on the horizon
VIEW_MARGIN_DEG = 180#5.0      # also draw satellites this far outside the fixed view
COORD_DECIMALS = 2         # lat/lon rounding in the HTML (0.01° ≈ 1 km)

# Legend groups that the "Broadband only / All satellites" switch hides or shows
TOGGLE_GROUPS = ("other", "inactive")


# ---- small helpers -------------------------------------------------------------

def _segments(lat_a, lon_a, lat_b, lon_b):
    """Line segments a -> b as one trace, separated by gaps (NaN)."""
    gap = np.full(len(lat_a), np.nan)
    lat = np.column_stack([lat_a, lat_b, gap]).ravel()
    lon = np.column_stack([lon_a, lon_b, gap]).ravel()
    return lat, lon


def _utc(unix_s):
    return pd.Timestamp(int(unix_s), unit="s", tz="UTC").strftime("%H:%M:%S")


def _index_pairs(pairs, index_a, index_b=None):
    """[[a, b], ...] -> (n, 2) array of indices. Pairs with an unknown name are skipped."""
    index_b = index_a if index_b is None else index_b
    rows = [[index_a[a], index_b[b]] for a, b in (pairs or [])
            if a in index_a and b in index_b]
    return np.array(rows, dtype=int).reshape(-1, 2)


def _per_step(sc, links, name):
    """Check that `links` has one entry per time step of sc (None is allowed)."""
    if links is not None and len(links) != len(sc):
        raise ValueError(f"{name} has {len(links)} time steps, the scenario has {len(sc)}")
    return links


def _locations(table, names):
    """{name: (lat, lon)} for the rows of a gateway or PoP table."""
    return {n: (la, lo) for n, la, lo in zip(names, table["lat"], table["lon"])}


def _shell_color(k):
    return SHELL_COLORS[k % len(SHELL_COLORS)]


# ---- shells and planes ---------------------------------------------------------

def _shell_of_satellites(names, shells, sat_name_to_shell):
    """Per satellite (in `names` order): index of its shell in `shells`, -1 if none.

    Uses sat_name_to_shell ({name: shell label}) if given, so satellites of a
    shell that are in no plane still count; otherwise the shells' plane members.
    """
    if sat_name_to_shell is None:
        sat_name_to_shell = {n: shell.label for shell in shells
                             for plane in shell.planes for n in plane.sat_names}
    shell_index = {shell.label: k for k, shell in enumerate(shells)}
    return np.array([shell_index.get(sat_name_to_shell.get(n), -1) for n in names],
                    dtype=int)


def _isl_shell(isl_idx, shell_of):
    """Per ISL: the shell of both satellites, -1 if they are in different/no shells."""
    a, b = shell_of[isl_idx[:, 0]], shell_of[isl_idx[:, 1]]
    return np.where(a == b, a, -1)


def _sat_info(names, satrec_by_name, version_by_name, shells, shell_of):
    """Per satellite: 'v2 mini · incl 53.16°<br>53.16° / 463 km · plane 17'."""
    plane_of = {n: plane.index for shell in shells
                for plane in shell.planes for n in plane.sat_names}
    info = []
    for name, k in zip(names, shell_of):
        satrec = satrec_by_name.get(name)
        incl = f"incl {np.degrees(satrec.inclo):.2f}°" if satrec is not None else "incl ?"
        text = f"{version_by_name.get(name, 'version ?')} · {incl}"
        if k >= 0:
            text += f"<br>{shells[k].label}"
            if name in plane_of:
                text += f" · plane {plane_of[name]}"
        info.append(text)
    return np.array(info, dtype=object)


def _orbit_circles(xyz, planes, sat_index, n_points):
    """Lat/lon of the orbit of every plane at one moment, as one line trace.

    All satellites of a plane lie in one plane through Earth's centre; its
    intersection with the globe is a great circle, found from a plane fit
    (SVD) through the satellites' ECEF positions. Planes with fewer than two
    satellites are skipped.
    """
    u = np.linspace(0.0, 2.0 * np.pi, n_points + 1)
    lats, lons = [], []
    for plane in planes:
        idx = [sat_index[n] for n in plane.sat_names if n in sat_index]
        points = xyz[idx]
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 2:
            continue
        _, _, vt = np.linalg.svd(points, full_matrices=False)
        circle = np.outer(np.cos(u), vt[0]) + np.outer(np.sin(u), vt[1])
        lats += [np.degrees(np.arcsin(np.clip(circle[:, 2], -1, 1))), [np.nan]]
        lons += [np.degrees(np.arctan2(circle[:, 1], circle[:, 0])), [np.nan]]
    if not lats:
        return np.array([]), np.array([])
    return np.concatenate(lats), np.concatenate(lons)


def _orbit_traces(sc, shells, sat_index, n_points):
    """One static, initially hidden orbit trace per shell (middle of the window)."""
    mid = len(sc) // 2
    traces = []
    for k, shell in enumerate(shells):
        lat, lon = _orbit_circles(sc.sat_xyz[mid], shell.planes, sat_index, n_points)
        traces.append(go.Scattergeo(
            lat=lat, lon=lon, mode="lines", hoverinfo="skip",
            line=dict(width=1, color=_shell_color(k)), opacity=0.5,
            name=f"plane orbits ({len(shell.planes)})", legendgroup=f"shell{k}",
            legendgrouptitle_text=shell.label, visible="legendonly"))
    return traces


# ---- comparison paths ----------------------------------------------------------

def _path_traces(row, node_latlon):
    """Traces for the compared path and (if different) the nearest-PoP path,
    plus a summary for the aircraft hover. Empty traces if this time step was
    not part of the comparison."""
    def polyline(path):
        points = [node_latlon(node) for node in path]
        return [p[0] for p in points], [p[1] for p in points], list(path)

    path = list(row["path"]) if row else []
    nearest = list(row["nearest_path"]) if row else []
    if nearest == path:
        nearest = []

    summary = ""
    if row:
        summary = (f"<br><b>comparison</b> → {row['pop']}: "
                   f"est {row['estimated_rtt_ms']:.1f} ms, meas {row['measured_rtt_ms']:.1f} ms, "
                   f"{row['isl_hops']} ISL hops"
                   f"<br>nearest PoP {row['nearest_pop']}: est {row['nearest_pop_rtt_ms']:.1f} ms")

    lat, lon, text = polyline(path)
    if text:
        text[0] = "aircraft" + summary                  # start point shows the summary
    used = go.Scattergeo(lat=lat, lon=lon, text=text, hoverinfo="text",
                         mode="lines+markers",
                         line=dict(width=4, color=COLORS["path"]),
                         marker=dict(size=6, color=COLORS["path"]),
                         name="Path used for comparison (measured PoP)",
                         legendgroup="path")

    lat, lon, text = polyline(nearest)
    other = go.Scattergeo(lat=lat, lon=lon, text=text, hoverinfo="text",
                          mode="lines+markers",
                          line=dict(width=2, color=COLORS["path"], dash="dash"),
                          marker=dict(size=4, color=COLORS["path"]), opacity=0.6,
                          name="Path to nearest PoP (if different)",
                          legendgroup="path_nearest")
    return [used, other], summary


# ---- traces per frame ------------------------------------------------------------

@dataclass
class _FigureData:
    """Everything the frames share, so _frame_traces needs few arguments."""
    sc: object
    names: np.ndarray          # satellite names, in sc order
    sat_lat: np.ndarray        # (time, satellite)
    sat_lon: np.ndarray
    sat_alt_km: np.ndarray
    sat_info: np.ndarray       # static hover text per satellite
    shells: list
    shell_of: np.ndarray       # shell index per satellite, -1 if none
    intra: np.ndarray          # (n, 2) satellite indices of intra-plane ISLs
    intra_shell: np.ndarray    # shell per intra-plane ISL, -1 if none/between shells
    inter: np.ndarray          # (n, 2) satellite indices of inter-plane ISLs
    inter_shell: np.ndarray
    gw_lat: np.ndarray
    gw_lon: np.ndarray
    inactive: np.ndarray       # bool per satellite: non-operational
    bb: np.ndarray             # bool per satellite: operating broadband
    reasons: dict              # {name: why non-operational}
    view_centre: np.ndarray    # ECEF unit vector of the fixed view's centre
    view_radius_deg: float     # satellites farther from the centre are not drawn


def _frame_traces(i, d, feeder, linked_idx, path_traces, path_summary):
    """Traces for time step i, always in the same order (frames rely on it)."""
    names = d.names
    elev = d.sc.sat_elev_from_aircraft[i]
    lat, lon, alt_km = d.sat_lat[i], d.sat_lon[i], d.sat_alt_km[i]
    lat_ac, lon_ac, alt_ac = d.sc.aircraft_lla[i]

    linked = np.zeros(len(names), dtype=bool)
    linked[linked_idx] = True
    xyz = d.sc.sat_xyz[i]
    from_centre = np.degrees(np.arccos(np.clip(
        xyz @ d.view_centre / np.linalg.norm(xyz, axis=1), -1, 1)))
    in_view = from_centre < d.view_radius_deg           # only these are drawn
    bb_in_view = in_view & d.bb & ~d.inactive & ~linked
    other = in_view & ~d.bb & ~d.inactive & ~linked
    inactive = in_view & d.inactive & ~linked

    def hover(mask, note):
        texts = []
        for name, info, alt, e in zip(names[mask], d.sat_info[mask], alt_km[mask], elev[mask]):
            text = f"<b>{name}</b><br>{info}<br>alt {alt:.0f} km · elev {e:.1f}°"
            if note == "linked":
                text += "<br>linked to aircraft"
            if note == "inactive":
                text += "<br>non-operational" + (f": {d.reasons[name]}" if name in d.reasons else "")
            texts.append(text)
        return texts

    def markers(mask, name, marker, group, note=None, title=None):
        return go.Scattergeo(lat=lat[mask], lon=lon[mask], text=hover(mask, note),
                             hoverinfo="text", mode="markers", marker=marker,
                             name=name, legendgroup=group, legendgrouptitle_text=title)

    def lines(seg_lat, seg_lon, name, color, group, width=1.5, opacity=0.7,
              dash=None, title=None):
        return go.Scattergeo(lat=seg_lat, lon=seg_lon, mode="lines", hoverinfo="skip",
                             line=dict(width=width, color=color, dash=dash),
                             opacity=opacity, name=name, legendgroup=group,
                             legendgrouptitle_text=title)

    def isl_segments(pairs, keep):
        """Segments of the selected ISLs whose two satellites are both in view."""
        a, b = pairs[keep, 0], pairs[keep, 1]
        both = in_view[a] & in_view[b]
        a, b = a[both], b[both]
        return _segments(lat[a], lon[a], lat[b], lon[b])

    # ISLs and satellites outside the shells (all of them if no shells are given)
    no_shell = [
        lines(*isl_segments(d.intra, d.intra_shell == -1),
              "Laser ISL, intra-plane", COLORS["isl_intra"], "isl_intra"),
        lines(*isl_segments(d.inter, d.inter_shell == -1),
              "Laser ISL, inter-plane", COLORS["isl_inter"], "isl_inter", dash="dash"),
    ]

    # Feeder links of this time step whose satellite is in view
    g, s = feeder[:, 0], feeder[:, 1]
    g, s = g[in_view[s]], s[in_view[s]]
    feeder_lines = lines(*_segments(d.gw_lat[g], d.gw_lon[g], lat[s], lon[s]),
                         "Gateway–satellite feeder link", COLORS["feeder"], "feeder",
                         opacity=0.8, dash="dot")

    no_shell_sats = markers(bb_in_view & (d.shell_of == -1),
                            "Broadband satellite (no shell)" if d.shells
                            else "Broadband satellite in view",
                            dict(size=5, color=COLORS["sat_horizon"]), "bb")

    # One legend group per shell: satellites, intra-plane ISLs, inter-plane ISLs
    shell_traces = []
    for k, shell in enumerate(d.shells):
        group, title = f"shell{k}", shell.label
        shell_traces += [
            markers(bb_in_view & (d.shell_of == k), "satellites",
                    dict(size=5, color=_shell_color(k)), group, title=title),
            lines(*isl_segments(d.intra, d.intra_shell == k), "intra-plane ISLs",
                  COLORS["isl_intra"], group, title=title),
            lines(*isl_segments(d.inter, d.inter_shell == k), "inter-plane ISLs",
                  COLORS["isl_inter"], group, dash="dash", title=title),
        ]

    # Aircraft -> linked satellites
    n = int(linked.sum())
    aircraft_lines = lines(*_segments(np.full(n, lat_ac), np.full(n, lon_ac),
                                      lat[linked], lon[linked]),
                           "Aircraft–satellite link", COLORS["sat_linked"], "linked",
                           width=2, opacity=0.8)

    def n_isl_in_view(pairs):
        return int((in_view[pairs[:, 0]] & in_view[pairs[:, 1]]).sum())

    aircraft_text = (f"{_utc(d.sc.unix_s[i])} UTC<br>alt {alt_ac:,.0f} m<br>"
                     f"linked to: {', '.join(names[linked]) or 'none'}<br>"
                     f"{int(((elev > 0) & d.bb & ~d.inactive).sum())} broadband sats above horizon<br>"
                     f"{int(other.sum())} other operating sats in view<br>"
                     f"{int(inactive.sum())} non-operational in view<br>"
                     f"{n_isl_in_view(d.intra)} intra-plane / "
                     f"{n_isl_in_view(d.inter)} inter-plane ISLs in view<br>"
                     f"{len(s)} feeder links in view"
                     f"{path_summary}")
    return [
        *no_shell,
        feeder_lines,
        no_shell_sats,
        *shell_traces,
        aircraft_lines,
        markers(linked, "Satellite linked to aircraft",
                dict(size=10, color=COLORS["sat_linked"], line=dict(width=1, color="white")),
                "linked", note="linked"),
        markers(other, "Other operating satellite (e.g. direct-to-cell)",
                dict(size=5, color=COLORS["sat_other"]), "other"),
        markers(inactive, "Non-operational satellite",
                dict(size=7, symbol="x", color=COLORS["sat_inactive"]),
                "inactive", note="inactive"),
        *path_traces,
        go.Scattergeo(lat=[lat_ac], lon=[lon_ac], mode="markers",
                      text=[aircraft_text], hoverinfo="text",
                      marker=dict(size=13, symbol="diamond", color=COLORS["aircraft"],
                                  line=dict(width=2, color="white")),
                      name="Aircraft", legendgroup="aircraft"),
    ]


# ---- static traces and view ------------------------------------------------------

def _ground_traces(sc, ground_links):
    """Static lines for [[pop, pop], ..., [gateway, pop], ...]: backbone and backhaul."""
    pops = _locations(sc.pops, sc.pops["code"])
    location = {**_locations(sc.gateways, sc.gateways["name"]), **pops}

    backbone, backhaul = [], []
    for a, b in ground_links:
        if a in location and b in location:
            (backbone if a in pops and b in pops else backhaul).append((a, b))

    def trace(links, name, color, width, opacity, dash=None):
        lat_a, lon_a = np.array([location[a] for a, _ in links]).reshape(-1, 2).T
        lat_b, lon_b = np.array([location[b] for _, b in links]).reshape(-1, 2).T
        lat, lon = _segments(lat_a, lon_a, lat_b, lon_b)
        return go.Scattergeo(lat=lat, lon=lon, mode="lines", hoverinfo="skip",
                             line=dict(width=width, color=color, dash=dash),
                             opacity=opacity, name=f"{name} ({len(links)})",
                             legendgroup=name)

    return [
        trace(backbone, "PoP backbone", COLORS["backbone"], 1.5, 0.6),
        trace(backhaul, "Gateway–PoP backhaul", COLORS["backhaul"], 1, 0.5, dash="dash"),
    ]


def _static_traces(sc, ground_links, orbit_traces):
    track, gw, pops = sc.flight_track, sc.gateways, sc.pops
    ground = _ground_traces(sc, ground_links) if ground_links else []
    return orbit_traces + ground + [
        go.Scattergeo(lat=track["lat"], lon=track["lon"], mode="lines",
                      line=dict(width=2, color=COLORS["track"]), hoverinfo="skip",
                      name="Flight track"),
        go.Scattergeo(lat=pops["lat"], lon=pops["lon"], mode="markers",
                      text=pops["code"] + "<br>" + pops["city"] + ", " + pops["country"],
                      hoverinfo="text",
                      marker=dict(size=9, symbol="square", color=COLORS["pop"]),
                      name="PoP"),
        go.Scattergeo(lat=gw["lat"], lon=gw["lon"], mode="markers",
                      text=gw["name"] + "<br>" + gw.get("status", ""), hoverinfo="text",
                      marker=dict(size=8, symbol="triangle-up", color=COLORS["gateway"]),
                      name="Gateway"),
    ]


def _fixed_view(sc):
    """One view for the whole animation: centred on the window's track and
    zoomed so every aircraft position plus its satellite horizon fits.

    Returns (projection settings, view centre as ECEF unit vector, view radius in deg)."""
    centre = sc.aircraft_xyz.mean(axis=0)
    centre /= np.linalg.norm(centre)
    lat, lon, _ = ecef_to_lla(centre * 6_371_000.0)

    unit = sc.aircraft_xyz / np.linalg.norm(sc.aircraft_xyz, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(unit @ centre, -1, 1))).max()
    radius = min(spread + HORIZON_DEG, 90.0)
    projection = dict(rotation=dict(lat=float(lat), lon=float(lon)),
                      scale=float(1.0 / np.sin(np.radians(radius))))
    return projection, centre, radius


# ---- figure ----------------------------------------------------------------------

def make_figure(sc, isl_pairs=None, inter_isl_pairs=None, non_operational=None,
                sat_feeder_links=None, ground_links=None, aircraft_links=None,
                comparison=None, satrec_by_name=None, version_by_name=None,
                bb_operating_satrec_by_name=None, bb_only_at_start=False,
                shells=None, sat_name_to_shell=None, orbit_points=72,
                frame_step=1, frame_ms=100):
    """Build the animated Plotly figure.

    The view is fixed for the whole animation (see `_fixed_view`); zooming or
    panning by hand is kept while it plays.

    isl_pairs:       [(name_a, name_b), ...] intra-plane laser links (from
                     find_intra_isl_pairs); None or [] draws none.
    inter_isl_pairs: [(name_a, name_b), ...] inter-plane laser links; None or
                     [] draws none. ISLs are drawn only when both satellites
                     are inside the view.
    non_operational: satellite names to mark as non-operational (list or set),
                     or a {name: reason} dict to also show the reason on hover.
    sat_feeder_links: one list per time step of sc, each [[gateway, satellite], ...]
                     (from find_feeder_links); None draws no feeder links.
    ground_links:    [[pop, pop], ..., [gateway, pop], ...] (from
                     find_ground_links); drawn as static lines. None draws none.
    aircraft_links:  one list per time step of sc, each [[aircraft, satellite], ...]
                     (from find_aircraft_links); linked satellites are drawn in
                     blue with a line to the aircraft. None draws no links.
    comparison:      the table returned by compare(); its paths are drawn in
                     the time steps it covers. None draws no paths.
    satrec_by_name:  {name: Satrec}, used for the inclination on hover.
    version_by_name: {name: version}, shown on hover. Include all satellites,
                     not only the operational ones.
    bb_operating_satrec_by_name:
                     the operating broadband satellites (only the keys, i.e.
                     the names, are used). Operating satellites not in it are
                     drawn as "other". None treats every operating satellite
                     as broadband.
    bb_only_at_start: start with other and non-operational satellites hidden.
    shells:          Shell objects with planes filled in (shell.planes, each
                     Plane with index and sat_names), e.g. bb_shells. Every
                     shell gets its own legend group (satellites, intra-plane
                     ISLs, inter-plane ISLs, plane orbits), each item
                     switchable in the legend; orbits start hidden. Shells are
                     told apart by shell.label, so labels must be unique.
                     None draws everything without shell groups.
    sat_name_to_shell: {name: shell label} (from find_shells_dbscan). Makes
                     shell members that are in no plane count for their shell
                     too. None uses the plane members only.
    orbit_points:    points per drawn plane orbit.
    frame_step:      draw every n-th second (data stays at 1 s; this only thins
                     the animation for long windows).
    frame_ms:        playback time per frame in milliseconds.
    """
    shells = shells or []
    names = np.asarray(sc.sat_names)
    sat_index = {name: k for k, name in enumerate(names)}
    gw_index = {name: k for k, name in enumerate(sc.gateways["name"])}
    non_operational = non_operational or []
    sat_lat, sat_lon, sat_alt = sc.sat_lla()
    sat_lat, sat_lon = np.round(sat_lat, COORD_DECIMALS), np.round(sat_lon, COORD_DECIMALS)
    projection, view_centre, view_radius = _fixed_view(sc)

    shell_of = _shell_of_satellites(names, shells, sat_name_to_shell)
    intra = _index_pairs(isl_pairs, sat_index)
    inter = _index_pairs(inter_isl_pairs, sat_index)
    d = _FigureData(
        sc=sc, names=names, sat_lat=sat_lat, sat_lon=sat_lon,
        sat_alt_km=sat_alt / 1000.0,                     # sat_lla altitude is in metres
        sat_info=_sat_info(names, satrec_by_name or {}, version_by_name or {},
                           shells, shell_of),
        shells=shells, shell_of=shell_of,
        intra=intra, intra_shell=_isl_shell(intra, shell_of),
        inter=inter, inter_shell=_isl_shell(inter, shell_of),
        gw_lat=sc.gateways["lat"].to_numpy(), gw_lon=sc.gateways["lon"].to_numpy(),
        inactive=np.isin(names, list(non_operational)),
        bb=(np.ones(len(names), dtype=bool) if bb_operating_satrec_by_name is None
            else np.isin(names, list(bb_operating_satrec_by_name))),
        reasons=non_operational if isinstance(non_operational, dict) else {},
        view_centre=view_centre, view_radius_deg=view_radius + VIEW_MARGIN_DEG,
    )

    feeder_per_step = _per_step(sc, sat_feeder_links, "sat_feeder_links")
    aircraft_per_step = _per_step(sc, aircraft_links, "aircraft_links")
    comparison_rows = ({} if comparison is None else
                       {int(round(row["unix_s"])): row for row in comparison.to_dict("records")})
    ground_nodes = {**{f"gw:{n}": p for n, p in _locations(sc.gateways, sc.gateways["name"]).items()},
                    **{f"pop:{n}": p for n, p in _locations(sc.pops, sc.pops["code"]).items()}}

    static = _static_traces(sc, ground_links, _orbit_traces(sc, shells, sat_index, orbit_points))

    frame_data = []
    for i in range(0, len(sc), frame_step):
        def node_latlon(node):
            """(lat, lon) of a graph node ID at time step i; NaN if unknown."""
            if node == "aircraft":
                return sc.aircraft_lla[i][0], sc.aircraft_lla[i][1]
            if node.startswith("sat:") and node[4:] in sat_index:
                k = sat_index[node[4:]]
                return sat_lat[i, k], sat_lon[i, k]
            return ground_nodes.get(node, (np.nan, np.nan))

        feeder = _index_pairs(feeder_per_step[i] if feeder_per_step is not None else None,
                              gw_index, sat_index)
        linked_idx = np.array([sat_index[s] for _, s in aircraft_per_step[i] if s in sat_index]
                              if aircraft_per_step is not None else [], dtype=int)
        row = comparison_rows.get(int(round(sc.unix_s[i])))
        path_traces, path_summary = _path_traces(row, node_latlon)
        frame_data.append((_utc(sc.unix_s[i]),
                           _frame_traces(i, d, feeder, linked_idx, path_traces, path_summary)))

    # Frames replace only the dynamic traces, which come after the static ones
    first = frame_data[0][1]
    dynamic_ids = list(range(len(static), len(static) + len(first)))
    frames = [go.Frame(name=name, traces=dynamic_ids, data=data) for name, data in frame_data]
    steps = [dict(label=name, method="animate",
                  args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))])
             for name, _ in frame_data]

    fig = go.Figure(data=static + first, frames=frames)

    # Traces hidden by "Broadband only"; visibility persists through all frames
    toggle_ids = [len(static) + k for k, trace in enumerate(first)
                  if trace.legendgroup in TOGGLE_GROUPS]
    if bb_only_at_start:
        for k in toggle_ids:
            fig.data[k].visible = "legendonly"

    play = dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True,
                transition=dict(duration=0), mode="immediate")
    start = pd.Timestamp(int(sc.unix_s[0]), unit="s", tz="UTC")
    fig.update_layout(
        title=dict(text=f"Aircraft and Starlink satellites — {start:%Y-%m-%d %H:%M:%S} UTC, "
                        f"{len(sc)} s at 1 s resolution, "
                        f"{int((d.bb & ~d.inactive).sum())} operating broadband satellites",
                   x=0.02),
        height=800, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(252,252,251,0.85)",
                    groupclick="toggleitem"),
        uirevision="fixed",
        geo=dict(projection=dict(type="orthographic", **projection),
                 showland=True, landcolor="#ecebe7", showocean=True, oceancolor="#f6f6f4",
                 showcountries=True, countrycolor="#c9c8c2", coastlinecolor="#a3a29c",
                 showlakes=False, lataxis_showgrid=True, lonaxis_showgrid=True,
                 lataxis_gridcolor="#e2e1dc", lonaxis_gridcolor="#e2e1dc"),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.02, y=0.06,
                 buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                          dict(label="❚❚ Pause", method="animate",
                               args=[[None], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=False))])]),
            dict(type="buttons", direction="left", x=0.99, xanchor="right",
                 y=0.98, yanchor="top", active=1 if bb_only_at_start else 0,
                 buttons=[dict(label="All satellites", method="restyle",
                               args=[{"visible": True}, toggle_ids]),
                          dict(label="Broadband only", method="restyle",
                               args=[{"visible": "legendonly"}, toggle_ids])]),
        ],
        sliders=[dict(steps=steps, x=0.15, len=0.83, y=0.08, ticklen=0,
                      currentvalue=dict(prefix="UTC ", font=dict(size=14)),
                      font=dict(size=1, color="rgba(0,0,0,0)"))],
    )
    return fig


def save_html(sc, non_operational, isl_pairs, path, **kwargs):
    make_figure(sc, isl_pairs=isl_pairs, non_operational=non_operational,
                **kwargs).write_html(path, include_plotlyjs="cdn", auto_play=False)
    return path