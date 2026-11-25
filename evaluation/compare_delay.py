"""
compare_delay.py

Compare measured RTTs with a propagation-only RTT floor from the network graph.

For every time step in the interval:
  * measured: mean of all RTT measurements falling into that time step
  * PoP: the PoP code of the measurement (source_pop). Steps without a
    measurement use the last measured PoP (the next one at the start).
  * estimated: 2 x the one-way propagation delay of the fastest path from the
    aircraft to that PoP. Space links (ISL, feeder, aircraft) carry light at c,
    ground links carry it in fibre at c / 1.468. Processing and queueing are
    not included, so this is a lower bound on the RTT.
  * reference: the same for the fastest path to any PoP.

The paths (node IDs from the aircraft to the PoP) are kept in the result, so
the viewer can draw them (save_html(..., comparison=result)).

PoP nodes in the graph are named "pop:<code>", the same codes as in the
measurements, so no lookup table is needed.

Usage:
    result = compare(sc.unix_s, (start, end), tg, "outside_in_ping.csv")
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SPEED_OF_LIGHT_KM_S = 299_792.458
FIBRE_REFRACTIVE_INDEX = 1.468               # light in fibre travels at about c / 1.47
SPACE_LINK_TYPES = {"isl", "feeder", "aircraft"}
AIRCRAFT = "aircraft"
UNIX_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def compare(unix_s, time_interval, tg, delay_meas_csv,
            out_png="delay_comparison.png", every=1):
    """Plot measured RTT vs the estimated RTT floor and return a table per time step.

    unix_s         -- simulation times, e.g. sc.unix_s
    time_interval  -- (start, end) as Unix seconds or ISO strings, inside unix_s
    tg             -- TemporalGraph from build_temporal_graph
    delay_meas_csv -- measurement log with columns timestamp, rtt, pop, source_pop
    out_png        -- where to save the plot
    every          -- evaluate only every n-th time step (faster for long intervals)
    """
    unix_s = np.asarray(unix_s, dtype=float)
    steps = steps_in_interval(unix_s, time_interval)[::every]
    measured = measured_rtt_per_step(delay_meas_csv, unix_s).reindex(steps)
    pop_per_step = measured["measured_pop"].ffill().bfill()

    rows = []
    for i in steps:
        G = tg.graph(i)
        pop = pop_per_step.loc[i]
        estimated_rtt_ms, path = rtt_to_pop(G, pop)
        nearest_rtt_ms, nearest_pop, nearest_path = rtt_to_nearest_pop(G)
        path_types = link_types(G, path)
        rows.append(dict(
            unix_s=unix_s[i],
            measured_rtt_ms=measured.loc[i, "measured_rtt_ms"],
            n_measurements=0 if pd.isna(measured.loc[i, "n_measurements"])
                           else int(measured.loc[i, "n_measurements"]),
            pop=pop,
            pop_measured_in_step=pd.notna(measured.loc[i, "measured_pop"]),
            estimated_rtt_ms=estimated_rtt_ms,
            path=path,
            path_types=path_types,
            isl_hops=path_types.count("isl"),
            nearest_pop=nearest_pop,
            nearest_pop_rtt_ms=nearest_rtt_ms,
            nearest_path=nearest_path,
        ))
    result = pd.DataFrame(rows)

    plot_comparison(result, out_png)
    print_summary(result)
    return result


# ---- estimated RTT floor -----------------------------------------------------

def one_way_delay_ms(u, v, link):
    """Propagation delay of one link, from its length and medium."""
    speed = SPEED_OF_LIGHT_KM_S
    if link["type"] not in SPACE_LINK_TYPES:
        speed /= FIBRE_REFRACTIVE_INDEX
    return link["distance_km"] / speed * 1000.0


def link_types(G, path):
    return [G.edges[a, b]["type"] for a, b in zip(path, path[1:])]


def rtt_to_pop(G, pop_code):
    """(RTT floor in ms, path) for the fastest path aircraft -> pop:<code>.

    The path is a list of node IDs from the aircraft to the PoP;
    (nan, []) if the PoP is unknown or unreachable.
    """
    target = f"pop:{pop_code}"
    if pd.isna(pop_code) or target not in G:
        return np.nan, []
    try:
        one_way_ms, path = nx.single_source_dijkstra(G, AIRCRAFT, target,
                                                     weight=one_way_delay_ms)
    except nx.NetworkXNoPath:
        return np.nan, []
    return 2.0 * one_way_ms, path


def rtt_to_nearest_pop(G):
    """(RTT floor in ms, PoP code, path) for the fastest path from the aircraft
    to any PoP. The path runs from the aircraft to the PoP; (nan, None, [])
    if there is none."""
    pops = [n for n, node_type in G.nodes(data="type") if node_type == "pop"]
    try:
        one_way_ms, path = nx.multi_source_dijkstra(G, pops, target=AIRCRAFT,
                                                    weight=one_way_delay_ms)
    except nx.NetworkXNoPath:
        return np.nan, None, []
    path = path[::-1]                                  # aircraft -> PoP
    return 2.0 * one_way_ms, path[-1].removeprefix("pop:"), path


# ---- measurements --------------------------------------------------------------

def measured_rtt_per_step(csv_path, unix_s):
    """Mean RTT, number of measurements and PoP code per simulation time step.

    A measurement belongs to step i if unix_s[i] <= t < unix_s[i + 1].
    """
    log = pd.read_csv(csv_path)
    t = to_unix_seconds(log["timestamp"])
    step_length = np.median(np.diff(unix_s)) if len(unix_s) > 1 else 1.0

    step = np.searchsorted(unix_s, t, side="right") - 1
    inside = (step >= 0) & (t < unix_s[-1] + step_length)
    log = log[inside].assign(step=step[inside])

    return log.groupby("step").agg(measured_rtt_ms=("rtt", "mean"),
                                   n_measurements=("rtt", "size"),
                                   measured_pop=("source_pop", "first"))


def to_unix_seconds(timestamps):
    ts = pd.to_datetime(timestamps, utc=True)
    return ((ts - UNIX_EPOCH) / pd.Timedelta(seconds=1)).to_numpy()


def to_unix(value):
    """Unix seconds from a number or a date string (UTC if no zone is given)."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts.timestamp()


def steps_in_interval(unix_s, time_interval):
    start, end = (to_unix(x) for x in time_interval)
    if not unix_s[0] <= start <= end <= unix_s[-1]:
        raise ValueError(f"interval {start:.0f}–{end:.0f} is not inside the simulation "
                         f"{unix_s[0]:.0f}–{unix_s[-1]:.0f}")
    return np.where((unix_s >= start) & (unix_s <= end))[0]


# ---- output --------------------------------------------------------------------

def plot_comparison(result, out_png):
    time = pd.to_datetime(result["unix_s"], unit="s", utc=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time, result["measured_rtt_ms"], ".", markersize=3,
            label="Measured RTT (mean per time step)")
    ax.plot(time, result["estimated_rtt_ms"], "-", linewidth=1.5,
            label="Estimated RTT floor to the measured PoP")
    ax.plot(time, result["nearest_pop_rtt_ms"], "--", linewidth=0.8, color="grey",
            label="Estimated RTT floor to the nearest PoP")

    # Vertical line and code wherever the measured PoP changes
    changes = result["pop"] != result["pop"].shift()
    for t, pop in zip(time[changes], result["pop"][changes]):
        ax.axvline(t, color="black", linestyle=":", linewidth=0.8)
        ax.text(t, 1.0, f" {pop}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("RTT (ms)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def print_summary(result):
    both = result.dropna(subset=["measured_rtt_ms", "estimated_rtt_ms"])
    if both.empty:
        print("no time steps with both a measurement and an estimate")
        return
    gap = both["measured_rtt_ms"] - both["estimated_rtt_ms"]
    same_pop = (result["nearest_pop"] == result["pop"]).mean()
    print(f"{len(both)} time steps compared | measured - estimated: "
          f"median {gap.median():.1f} ms, min {gap.min():.1f} ms, max {gap.max():.1f} ms")
    print("measured PoPs:", result["pop"].value_counts().to_dict())
    print(f"nearest PoP = measured PoP in {same_pop:.0%} of time steps")
    unreachable = result["estimated_rtt_ms"].isna().sum()
    if unreachable:
        print(f"{unreachable} time steps without a path to the measured PoP")