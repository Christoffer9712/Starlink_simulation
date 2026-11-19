"""Animated globe of a Scenario: one animation frame per second of sim time.

Shown per frame
  * aircraft (moving) and its full recorded track
  * operating broadband satellites above the aircraft's horizon (grey) and
    above the minimum elevation, with a line to the aircraft (blue)
  * other operating satellites, e.g. direct-to-cell (amber)
  * non-operational satellites above the horizon (red ×; never linked)
  * laser ISLs between satellites, when at least one end is above the
    aircraft's horizon (purple; drawn between sub-satellite points)
  * gateway-satellite feeder links of this time step, when the satellite is
    above the aircraft's horizon (green, dotted)
Static: gateways, PoPs, the PoP backbone (dark) and gateway backhaul to
PoPs (grey, dashed).

A "Broadband only / All satellites" switch (top right) hides or shows the
other and non-operational satellites; their legend entries toggle them too.

Hovering over a satellite shows its name, version, inclination, current
altitude and elevation from the aircraft.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from positions_sim.geometry import ecef_to_lla

COLORS = {
    "aircraft": "#eb6834",
    "sat_usable": "#2a78d6",
    "sat_horizon": "#a3a29c",
    "sat_other": "#d9a441",
    "sat_inactive": "#d64545",
    "isl": "#8a4fc7",
    "feeder": "#1baf7a",
    "backbone": "#2b2b29",
    "backhaul": "#6b6a64",
    "gateway": "#1baf7a",
    "pop": "#2b2b29",
    "track": "#eb6834",
}

HORIZON_ELEV_DEG = -10.0   # satellites above this elevation are drawn at all
HORIZON_DEG = 23.0         # Earth-central angle to a 550 km satellite on the horizon

# Legend groups that the "Broadband only / All satellites" switch hides or shows
TOGGLE_GROUPS = ("other", "inactive")


def _segments(lat_a, lon_a, lat_b, lon_b):
    """Line segments a -> b as one trace, separated by gaps (NaN)."""
    gap = np.full(len(lat_a), np.nan)
    lat = np.column_stack([lat_a, lat_b, gap]).ravel()
    lon = np.column_stack([lon_a, lon_b, gap]).ravel()
    return lat, lon


def _pop_names(pops):
    """PoP names as used by find_ground_links: 'name' if present, else 'city'."""
    return list(pops["name"] if "name" in pops else pops["city"])


def _isl_indices(sc, isl_pairs):
    """Convert [[name_a, name_b], ...] to an (n, 2) array of satellite indices.

    Pairs with a satellite that is not in the scenario are skipped.
    """
    index = {name: k for k, name in enumerate(sc.sat_names)}
    rows = [[index[a], index[b]] for a, b in (isl_pairs or [])
            if a in index and b in index]
    return np.array(rows, dtype=int).reshape(-1, 2)


def _feeder_indices(sc, sat_feeder_links):
    """Per time step: (gateway rows, satellite columns) of the feeder links.

    sat_feeder_links must have one entry per time step of sc, each a list of
    [gateway_name, satellite_name]. Links with an unknown gateway or satellite
    are skipped. Returns None if no feeder links are given.
    """
    if sat_feeder_links is None:
        return None
    if len(sat_feeder_links) != len(sc):
        raise ValueError(f"sat_feeder_links has {len(sat_feeder_links)} time steps, "
                         f"the scenario has {len(sc)}")
    gw_index = {name: k for k, name in enumerate(sc.gateways["name"])}
    sat_index = {name: k for k, name in enumerate(sc.sat_names)}
    per_step = []
    for links in sat_feeder_links:
        rows = [[gw_index[g], sat_index[s]] for g, s in links
                if g in gw_index and s in sat_index]
        rows = np.array(rows, dtype=int).reshape(-1, 2)
        per_step.append((rows[:, 0], rows[:, 1]))
    return per_step


def _static_sat_info(sc, satrec_by_name, version_by_name):
    """Per satellite, in sc.sat_names order: 'v2 mini · incl 53.16°'."""
    satrec_by_name = satrec_by_name or {}
    version_by_name = version_by_name or {}
    info = []
    for name in sc.sat_names:
        version = version_by_name.get(name, "version ?")
        satrec = satrec_by_name.get(name)
        incl = f"incl {np.degrees(satrec.inclo):.2f}°" if satrec is not None else "incl ?"
        info.append(f"{version} · {incl}")
    return np.array(info, dtype=object)


def _dynamic_traces(i, sc, sat_lat, sat_lon, sat_alt_km, sat_info, isl_idx,
                    feeder, gw_lat, gw_lon, inactive_mask, bb_mask, reasons,
                    min_elev_deg):
    """Traces for time index i, in a fixed order (frames rely on it)."""
    elev = sc.sat_elev_from_aircraft[i]
    names = np.asarray(sc.sat_names)
    alt_km = sat_alt_km[i]
    lat_ac, lon_ac, alt_ac = sc.aircraft_lla[i]

    above = elev > HORIZON_ELEV_DEG
    high = elev >= min_elev_deg
    bb = bb_mask & ~inactive_mask
    other = ~bb_mask & ~inactive_mask

    bb_usable, other_usable = high & bb, high & other
    bb_horizon, other_horizon = above & bb & ~high, above & other & ~high
    inactive = above & inactive_mask

    def links_to(mask):
        n = int(mask.sum())
        return _segments(np.full(n, lat_ac), np.full(n, lon_ac),
                         sat_lat[i, mask], sat_lon[i, mask])

    bb_link_lat, bb_link_lon = links_to(bb_usable)
    other_link_lat, other_link_lon = links_to(other_usable)

    # Satellite <-> satellite laser links with at least one end in view
    a, b = isl_idx[:, 0], isl_idx[:, 1]
    shown = above[a] | above[b]
    a, b = a[shown], b[shown]
    isl_lat, isl_lon = _segments(sat_lat[i, a], sat_lon[i, a],
                                 sat_lat[i, b], sat_lon[i, b])

    # Gateway -> satellite feeder links of this time step, satellite in view
    g, s = feeder if feeder is not None else (np.array([], int), np.array([], int))
    in_view = above[s]
    g, s = g[in_view], s[in_view]
    feed_lat, feed_lon = _segments(gw_lat[g], gw_lon[g], sat_lat[i, s], sat_lon[i, s])

    def sat_hover(mask, non_operational=False):
        texts = []
        for nm, info, alt, e in zip(names[mask], sat_info[mask], alt_km[mask], elev[mask]):
            text = f"<b>{nm}</b><br>{info}<br>alt {alt:.0f} km · elev {e:.1f}°"
            if non_operational:
                text += "<br>non-operational" + (f": {reasons[nm]}" if nm in reasons else "")
            texts.append(text)
        return texts

    def markers(mask, name, marker, group, showlegend=True, non_operational=False):
        return go.Scattergeo(lat=sat_lat[i, mask], lon=sat_lon[i, mask],
                             text=sat_hover(mask, non_operational), hoverinfo="text",
                             mode="markers", marker=marker, name=name,
                             legendgroup=group, showlegend=showlegend)

    def lines(lat, lon, name, color, width, opacity, group, showlegend=True, dash=None):
        return go.Scattergeo(lat=lat, lon=lon, mode="lines", hoverinfo="skip",
                             line=dict(width=width, color=color, dash=dash),
                             opacity=opacity, name=name, legendgroup=group,
                             showlegend=showlegend)

    utc = pd.Timestamp(int(sc.unix_s[i]), unit="s", tz="UTC").strftime("%H:%M:%S")
    aircraft_text = (f"{utc} UTC<br>alt {alt_ac:,.0f} m<br>"
                     f"{int(bb_usable.sum())} usable broadband sats<br>"
                     f"{int(other_usable.sum())} other operating sats ≥ {min_elev_deg:g}°<br>"
                     f"{int(inactive.sum())} non-operational in view<br>"
                     f"{len(a)} ISLs in view<br>"
                     f"{len(s)} feeder links in view")
    return [
        lines(isl_lat, isl_lon, "Laser ISL (intra-plane)", COLORS["isl"], 1.5, 0.7, "isl"),
        lines(feed_lat, feed_lon, "Gateway–satellite feeder link", COLORS["feeder"],
              1.5, 0.8, "feeder", dash="dot"),
        markers(bb_horizon, "Broadband satellite above horizon",
                dict(size=5, color=COLORS["sat_horizon"]), "bb"),
        markers(bb_usable, f"Broadband satellite ≥ {min_elev_deg:g}° elevation",
                dict(size=8, color=COLORS["sat_usable"], line=dict(width=1, color="white")),
                "bb"),
        lines(bb_link_lat, bb_link_lon, "Aircraft–satellite link",
              COLORS["sat_usable"], 1, 0.5, "bb"),
        markers(other_horizon, "Other operating satellite (e.g. direct-to-cell)",
                dict(size=5, color=COLORS["sat_other"]), "other"),
        markers(other_usable, "", dict(size=8, color=COLORS["sat_other"],
                                       line=dict(width=1, color="white")),
                "other", showlegend=False),
        lines(other_link_lat, other_link_lon, "", COLORS["sat_other"], 1, 0.5,
              "other", showlegend=False),
        markers(inactive, "Non-operational satellite",
                dict(size=7, symbol="x", color=COLORS["sat_inactive"]),
                "inactive", non_operational=True),
        go.Scattergeo(lat=[lat_ac], lon=[lon_ac], mode="markers",
                      text=[aircraft_text], hoverinfo="text",
                      marker=dict(size=13, symbol="diamond", color=COLORS["aircraft"],
                                  line=dict(width=2, color="white")),
                      name="Aircraft", legendgroup="aircraft"),
    ]


def _ground_traces(sc, ground_links):
    """Static lines for [[pop, pop], ..., [gateway, pop], ...]: backbone and backhaul."""
    pop_names = _pop_names(sc.pops)
    location = {name: (lat, lon) for name, lat, lon
                in zip(sc.gateways["name"], sc.gateways["lat"], sc.gateways["lon"])}
    location.update({name: (lat, lon) for name, lat, lon
                     in zip(pop_names, sc.pops["lat"], sc.pops["lon"])})
    pop_set = set(pop_names)

    backbone, backhaul = [], []
    for a, b in (ground_links or []):
        if a in location and b in location:
            (backbone if a in pop_set and b in pop_set else backhaul).append((a, b))

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


def _static_traces(sc, ground_links=None):
    track, gw, pops = sc.flight_track, sc.gateways, sc.pops
    ground = _ground_traces(sc, ground_links) if ground_links else []
    return ground + [
        go.Scattergeo(lat=track["lat"], lon=track["lon"], mode="lines",
                      line=dict(width=2, color=COLORS["track"]), hoverinfo="skip",
                      name="Flight track"),
        go.Scattergeo(lat=pops["lat"], lon=pops["lon"], mode="markers",
                      text=pops["city"] + ", " + pops["country"], hoverinfo="text",
                      marker=dict(size=9, symbol="square", color=COLORS["pop"]),
                      name="PoP"),
        go.Scattergeo(lat=gw["lat"], lon=gw["lon"], mode="markers",
                      text=gw["name"] + "<br>" + gw.get("status", ""), hoverinfo="text",
                      marker=dict(size=8, symbol="triangle-up", color=COLORS["gateway"]),
                      name="Gateway"),
    ]


def _fixed_view(sc):
    """One view for the whole animation: centred on the window's track and
    zoomed so every aircraft position plus its satellite horizon fits."""
    centre = sc.aircraft_xyz.mean(axis=0)
    centre /= np.linalg.norm(centre)
    lat, lon, _ = ecef_to_lla(centre * 6_371_000.0)

    unit = sc.aircraft_xyz / np.linalg.norm(sc.aircraft_xyz, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(unit @ centre, -1, 1))).max()
    radius = min(spread + HORIZON_DEG, 90.0)
    return dict(rotation=dict(lat=float(lat), lon=float(lon)),
                scale=float(1.0 / np.sin(np.radians(radius))))


def make_figure(sc, isl_pairs=None, non_operational=None, sat_feeder_links=None,
                ground_links=None, satrec_by_name=None, version_by_name=None,
                bb_operating_satrec_by_name=None, bb_only_at_start=False,
                min_elev_deg=25.0, frame_step=1, frame_ms=100):
    """Build the animated Plotly figure.

    The view is fixed for the whole animation (see `_fixed_view`); zooming or
    panning by hand is kept while it plays.

    isl_pairs:       [[name_a, name_b], ...] laser links to draw (e.g. from
                     find_isl_pairs); None or [] draws no ISLs.
    non_operational: satellite names to mark as non-operational (list or set),
                     or a {name: reason} dict to also show the reason on hover.
                     These satellites are never counted as usable.
    sat_feeder_links: one list per time step of sc, each [[gateway, satellite], ...]
                     (from find_feeder_links); None draws no feeder links.
    ground_links:    [[pop, pop], ..., [gateway, pop], ...] (from
                     find_ground_links); drawn as static lines. None draws none.
    satrec_by_name:  {name: Satrec}, used for the inclination on hover.
    version_by_name: {name: version}, shown on hover. Include all satellites,
                     not only the operational ones.
    bb_operating_satrec_by_name:
                     the operating broadband satellites (only the keys, i.e.
                     the names, are used). Operating satellites not in it are
                     drawn as "other". None treats every operating satellite
                     as broadband.
    bb_only_at_start: start with other and non-operational satellites hidden.
    frame_step:      draw every n-th second (data stays at 1 s; this only thins
                     the animation for long windows).
    frame_ms:        playback time per frame in milliseconds.
    """
    sat_lat, sat_lon, sat_alt = sc.sat_lla()
    sat_alt_km = sat_alt / 1000.0                     # sat_lla altitude is in metres
    sat_info = _static_sat_info(sc, satrec_by_name, version_by_name)
    isl_idx = _isl_indices(sc, isl_pairs)
    feeder_idx = _feeder_indices(sc, sat_feeder_links)
    gw_lat = sc.gateways["lat"].to_numpy()
    gw_lon = sc.gateways["lon"].to_numpy()

    names = np.asarray(sc.sat_names)
    non_operational = non_operational or []
    reasons = non_operational if isinstance(non_operational, dict) else {}
    inactive_mask = np.isin(names, list(non_operational))
    bb_mask = (np.ones(len(names), dtype=bool) if bb_operating_satrec_by_name is None
               else np.isin(names, list(bb_operating_satrec_by_name)))

    static = _static_traces(sc, ground_links)

    frame_data = []
    for i in range(0, len(sc), frame_step):
        name = pd.Timestamp(int(sc.unix_s[i]), unit="s", tz="UTC").strftime("%H:%M:%S")
        feeder = feeder_idx[i] if feeder_idx is not None else None
        frame_data.append((name, _dynamic_traces(i, sc, sat_lat, sat_lon, sat_alt_km,
                                                 sat_info, isl_idx, feeder, gw_lat, gw_lon,
                                                 inactive_mask, bb_mask, reasons,
                                                 min_elev_deg)))

    # Frames replace only the dynamic traces, which come after the static ones
    first = frame_data[0][1]
    dynamic_ids = list(range(len(static), len(static) + len(first)))

    frames = [go.Frame(name=name, traces=dynamic_ids, data=data)
              for name, data in frame_data]
    steps = [dict(label=name, method="animate",
                  args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))])
             for name, _ in frame_data]

    fig = go.Figure(data=static + list(first), frames=frames)

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
                        f"{int((bb_mask & ~inactive_mask).sum())} operating broadband satellites",
                   x=0.02),
        height=800, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(252,252,251,0.85)"),
        uirevision="fixed",
        geo=dict(projection=dict(type="orthographic", **_fixed_view(sc)),
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