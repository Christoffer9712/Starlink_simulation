"""Render an animated globe of a flight and the Starlink satellites around it.

Example:
    python visualize_flight.py \
        --flight QTR68T.csv --tle starlink_3le.txt \
        --gateways starlink_gateways.kml --pops pops.json \
        --start 2026-08-21T00:47:00Z --duration 900 --out flight.html
"""
import argparse

import pandas as pd

from positions_sim import build_scenario
from visualize.visualize import save_html
from underlay_graph.isl_pairs import find_intra_isl_pairs
from collections import Counter
from underlay_graph.operational import stable_satrec, non_operational_reasons#, find_shells, find_planes, print_shells, mean_altitude_km, populate_shells
from underlay_graph.plane_detection import find_planes, plot_plane_raan
from underlay_graph.util import satversion
from simulation_param import LOW_ALTITUDE_KM, MIN_PLANE_SATELLITES
from underlay_graph.feeder_links import find_feeder_links
from underlay_graph.ground_links import find_ground_links, pop_labels
from underlay_graph.aircraft_links import find_aircraft_links
from underlay_graph.temporal_graph import build_temporal_graph
from evaluation.compare_delay import compare
from underlay_graph.shell_definitions import BB_SHELL_DEFINITIONS
from underlay_graph.shell_detection import find_shells_dbscan, print_shells, plot_shell_distribution

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flight", default='./data/QR720_41463d81.csv', help="flight trajectory CSV")
    p.add_argument("--delay_meas", default='./data/outside_in_ping_2605:59c0:700c:900::1_extracted.csv', help="flight delay data CSV")
    p.add_argument("--tle", default='./data/starlink_3le_2026_08_22.txt', help="Starlink 3-line TLE file")
    p.add_argument("--tle_week_prior", default='./data/starlink_3le_2026_08_15.txt', help="Starlink 3-line TLE file")
    p.add_argument("--tle_two_week_prior", default='./data/starlink_3le_2026_08_08.txt', help="Starlink 3-line TLE file")
    p.add_argument("--gateways", default='./data/starlink_gateways.kml', help="gateway KML")
    p.add_argument("--pops", default='./data/pops.json', help="PoP JSON")
    p.add_argument("--start", default='2026-08-21T05:47:00Z', help="window start, ISO UTC (default: first flight sample)")
    p.add_argument("--duration", type=int, default=40, help="window length in seconds")
    p.add_argument("--frame-step", type=int, default=5, help="animate every n-th second")
    p.add_argument("--frame-ms", type=int, default=1000, help="playback time per frame [ms]")
    p.add_argument("--out", default="flight_starlink.html")
    return p.parse_args()


def main():
    # Initialise scenarios
    a = parse_args()
    start = None if a.start is None else int(pd.Timestamp(a.start).timestamp())

    sc = build_scenario(a.flight, a.tle, a.gateways, a.pops,
                        start_unix=start, duration_s=a.duration)
    print(f"{len(sc)} time steps, {len(sc.sat_names)} satellites, "
          f"{len(sc.gateways)} gateways, {len(sc.pops)} PoPs")

    snapshot = sc.snapshot(0)
    sat_names = snapshot.sat_names
    satrec_by_name = snapshot.satrec_by_name
    when = pd.to_datetime(a.start, utc=True)

    version_by_name = {s: satversion(satrec_by_name[s]) for s in sat_names}

    # Find stable sats 
    sc_week_prior = build_scenario(a.flight, a.tle_week_prior, a.gateways, a.pops, start_unix=int(pd.Timestamp('2026-08-15T00:00:00Z').timestamp()), duration_s=1)
    sc_two_week_prior = build_scenario(a.flight, a.tle_two_week_prior, a.gateways, a.pops, start_unix=int(pd.Timestamp('2026-08-08T00:00:00Z').timestamp()), duration_s=1)
    satrec_by_name_week_prior = sc_week_prior.satrec_by_name
    satrec_by_name_two_week_prior = sc_two_week_prior.satrec_by_name

    # 1. Remove unstable satellites (altitude changes last two weeks)
    stable, unstable = stable_satrec(
        satrec_by_name, [satrec_by_name_week_prior, satrec_by_name_two_week_prior], max_change_km=10.0)
    why = {name: "unstable altitude" for name in unstable}

    # 2. Remove satellites with non-circular orbits, with SGP4 error, or when 2 sats share a slot
    why.update(non_operational_reasons(stable, when))

    # 3. One DBSCAN pass: noise = not in a shell
    operating_satrec_by_name = {n: s for n, s in stable.items() if n not in why}
    shells, noise, sat_name_to_shell = find_shells_dbscan(
        operating_satrec_by_name, version_by_name, eps=1.0, min_samples=70, inc_scale=0.1, alt_scale=5.0)

    # 3b. Split the mixed 53° / 463 km shell into sub-shells with finer scales
    mixed = next((s for s in shells
                  if round(s.inclination_deg) == 53 and 460 < s.altitude_km < 465), None)
    if mixed is not None:
        mixed_sats = {n: s for n, s in operating_satrec_by_name.items()
                      if sat_name_to_shell.get(n) == mixed.label}
        sub_shells, sub_noise, sub_sat_name_to_shell = find_shells_dbscan(
            mixed_sats, version_by_name, eps=1.0, min_samples=50, inc_scale=0.02, alt_scale=0.8)

        shells.remove(mixed)
        shells = sorted(shells + sub_shells, key=lambda s: (s.inclination_deg, s.altitude_km))
        noise += sub_noise
        for n in sub_noise:
            del sat_name_to_shell[n]               # no longer in any shell
        sat_name_to_shell.update(sub_sat_name_to_shell)

    why.update({n: "Not in shell" for n in noise})
    
    assert sum(s.n_satellites for s in shells) + len(why) == len(satrec_by_name)

    # 4. Remove shells below 400 km altitude as these are for DTC and not for broadband
    dtc_labels = {s.label for s in shells if s.altitude_km < 400}
    why.update({n: "DTC shell" for n, label in sat_name_to_shell.items() if label in dtc_labels})
    bb_shells = [s for s in shells if s.altitude_km >= 400]

    assert sum(s.n_satellites for s in bb_shells) + len(why) == len(satrec_by_name)

    bb_operating_satrec_by_name = {}
    for bb_shell in bb_shells:
        satrec_in_shell = {n: s for n, s in operating_satrec_by_name.items()
                        if sat_name_to_shell.get(n) == bb_shell.label}
        bb_operating_satrec_by_name.update(satrec_in_shell)
        planes, plane_by_name, unassigned = find_planes(satrec_in_shell, when, eps_deg=0.5, min_samples=3)
        bb_shell.planes = planes
        plot_plane_raan(bb_shell, f'RAAN_{int(bb_shell.altitude_km)}km_{int(bb_shell.inclination_deg)}deg.png')
        plot_shell_distribution(bb_shell, satrec_in_shell, version_by_name,
                                f"dist_{int(bb_shell.altitude_km)}km_{int(bb_shell.inclination_deg)}deg.png")
    print_shells(bb_shells)
    print(Counter(r.split(" (")[0] for r in why.values()))


    # Get ISLs
    intra_isl_pairs = find_intra_isl_pairs(bb_shells, bb_operating_satrec_by_name, version_by_name, when)
    
    print(f"{len(intra_isl_pairs)} intra-plane ISLs")


    sat_feeder_links = find_feeder_links(bb_operating_satrec_by_name, sc.gateways, sc.unix_s, max_per_gateway=8)
    ground_links = find_ground_links(sc.gateways, sc.pops)
    aircraft_links = find_aircraft_links(bb_operating_satrec_by_name, sc.sat_names, sc.sat_elev_from_aircraft, sc.unix_s, max_links=10)
    tg = build_temporal_graph(sc, aircraft_links, intra_isl_pairs, sat_feeder_links, ground_links)

    compare_time_interval = (sc.unix_s[0], sc.unix_s[-1])
    result = compare(sc.unix_s, compare_time_interval, tg, a.delay_meas, every=5, out_png="delay_comparison.png")

    save_html(sc, why, intra_isl_pairs, a.out,
        sat_feeder_links=sat_feeder_links,
        ground_links=ground_links,
        aircraft_links=aircraft_links,
        comparison=result,
        satrec_by_name=satrec_by_name,
        bb_operating_satrec_by_name=bb_operating_satrec_by_name,
        version_by_name=version_by_name,
        shells=bb_shells,
        frame_step=a.frame_step, frame_ms=a.frame_ms)
    
    print(f"wrote {a.out}")





    if False:
        # Automatically find shells or use predefined
        automatic = False #True
        if automatic:
            shells = find_shells(operating_satrec_by_name, version_by_name)
            # Broadband shells only (direct-to-cell shells sit below 400 km)
            bb_shells = [s for s in shells if s.altitude_km > 0] #LOW_ALTITUDE_KM]
            bb_plane_by_name = find_planes(bb_shells, operating_satrec_by_name, when)
        else:
            bb_shells, unassigned = populate_shells(BB_SHELL_DEFINITIONS,
                                                operating_satrec_by_name, version_by_name)
            bb_plane_by_name = find_planes(bb_shells, operating_satrec_by_name, when)
            print(f"{len(unassigned)} operating satellites fit no defined shell")

        s43 = next((s for s in bb_shells if s.label.startswith("43°")), None)
        if s43:
            plot_plane_raan(s43, operating_satrec_by_name, version_by_name, "shell_43deg_planes.png")

        # Satellites in defined shells only (replaces the altitude filter)
        bb_operating_satrec_by_name = {n: operating_satrec_by_name[n]
                                    for s in bb_shells for n in s.names}

        
        print_shells(bb_shells)          # after find_planes, so the planes column is filled
        for k, s in enumerate(bb_shells):
            print(k, f"{s.inclination_deg:.2f}° / {s.altitude_km:.0f} km, "
                    f"{s.n_satellites} sats, {len(s.planes)} planes")
        
        bb_operating_sats = list(bb_operating_satrec_by_name)
        bb_version_by_name = {s: satversion(bb_operating_satrec_by_name[s]) for s in bb_operating_sats}

        # Get ISLs
        intra_isl_pairs = find_isl_pairs(bb_shells, bb_operating_satrec_by_name, bb_version_by_name, when)
        inter_isl_pairs = find_inter_isl_pairs(bb_shells, bb_operating_satrec_by_name, bb_version_by_name, when)
        isl_pairs = intra_isl_pairs + inter_isl_pairs
        print(f"{len(intra_isl_pairs)} intra-plane and {len(inter_isl_pairs)} inter-plane ISLs")
        
        # Print some plane statistics
        # -----------------------------------------------------------------------------------
        linked = {n for pair in intra_isl_pairs for n in pair}
        for k, s in enumerate(bb_shells):
            sizes = Counter(len(p.names) for p in s.planes)
            unlinked = Counter()
            for p in s.planes:
                lasers = [n for n in p.names if has_laser_isl(version_by_name.get(n))]
                for n in lasers:
                    if n not in linked:
                        unlinked["plane too small" if len(p.names) < MIN_PLANE_SATELLITES
                                else "neighbours out of range"] += 1
            print(f"{s.inclination_deg:.2f}°/{s.altitude_km:.0f} km: {len(s.planes)} planes, "
                f"typical size {sizes.most_common(20)}, unlinked {dict(unlinked)}")
        # ------------------------------------------------------------------------------------

        # Optionally limit to the 8 highest satellites per gateway:
        sat_feeder_links = find_feeder_links(bb_operating_satrec_by_name, sc.gateways, sc.unix_s, max_per_gateway=8)

        ground_links = find_ground_links(sc.gateways, sc.pops)

        aircraft_links = find_aircraft_links(bb_operating_satrec_by_name, sc.sat_names, sc.sat_elev_from_aircraft, sc.unix_s, max_links=10)

        tg = build_temporal_graph(sc, aircraft_links, isl_pairs, sat_feeder_links, ground_links)

        compare_time_interval = (sc.unix_s[0], sc.unix_s[-1])
        result = compare(sc.unix_s, compare_time_interval, tg, a.delay_meas, every=5, out_png="delay_comparison.png")

        save_html(sc, why, intra_isl_pairs, a.out,
            inter_isl_pairs=inter_isl_pairs,
            sat_feeder_links=sat_feeder_links,
            ground_links=ground_links,
            aircraft_links=aircraft_links,
            comparison=result,
            satrec_by_name=satrec_by_name,
            bb_operating_satrec_by_name=bb_operating_satrec_by_name,
            version_by_name=version_by_name,
            shells=bb_shells,
            frame_step=a.frame_step, frame_ms=a.frame_ms)
        
        print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
