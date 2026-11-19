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
from underlay_graph.isl_pairs import find_isl_pairs, has_laser_isl
from collections import Counter
from underlay_graph.operational import non_operational_reasons
from underlay_graph.operational import find_shells, find_planes, print_shells, mean_altitude_km
from underlay_graph.util import satversion
from simulation_param import LOW_ALTITUDE_KM, MIN_PLANE_SATELLITES
from underlay_graph.feeder_links import find_feeder_links
from underlay_graph.ground_links import find_ground_links, pop_labels

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flight", default='./data/QR720_41463d81.csv', help="flight trajectory CSV")
    p.add_argument("--tle", default='./data/starlink_3le.txt', help="Starlink 3-line TLE file")
    p.add_argument("--gateways", default='./data/starlink_gateways.kml', help="gateway KML")
    p.add_argument("--pops", default='./data/pops.json', help="PoP JSON")
    p.add_argument("--start", default='2026-08-21T05:47:00Z', help="window start, ISO UTC (default: first flight sample)")
    p.add_argument("--duration", type=int, default=900, help="window length in seconds")
    p.add_argument("--min-elev", type=float, default=25.0, help="min elevation for a usable link [deg]")
    p.add_argument("--frame-step", type=int, default=1, help="animate every n-th second")
    p.add_argument("--frame-ms", type=int, default=100, help="playback time per frame [ms]")
    p.add_argument("--out", default="flight_starlink.html")
    return p.parse_args()


def main():
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

    #Remove non-operational satellites
    why = non_operational_reasons(sat_names, satrec_by_name)
    print(Counter(r.split(" (")[0] for r in why.values()))
    non_operational = set(why)

    operating_sats = [s for s in sat_names if s in satrec_by_name and s not in non_operational]
    operating_satrec_by_name = {s: satrec_by_name[s] for s in operating_sats}

    version_by_name = {s: satversion(satrec_by_name[s]) for s in operating_sats}
    print(Counter(version_by_name.values())) 

    # Print operational shells
    shells = find_shells(list(operating_satrec_by_name), operating_satrec_by_name,
                     version_by_name)

    # Broadband shells only (direct-to-cell shells sit below 400 km)
    bb_shells = [s for s in shells if s.altitude_km > LOW_ALTITUDE_KM]
    bb_plane_by_name = find_planes(bb_shells, operating_satrec_by_name, when)
    print_shells(bb_shells)          # after find_planes, so the planes column is filled

    # Remove satellites operating as DTC
    bb_operating_satrec_by_name = {s: operating_satrec_by_name[s] for s in operating_satrec_by_name if mean_altitude_km(operating_satrec_by_name[s])>LOW_ALTITUDE_KM}
    bb_operating_sats = list(bb_operating_satrec_by_name)
    bb_version_by_name = {s: satversion(bb_operating_satrec_by_name[s]) for s in bb_operating_sats}

    # Get ISLs
    intra_isl_pairs = find_isl_pairs(bb_shells, bb_operating_satrec_by_name, bb_version_by_name, when)
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
            f"typical size {sizes.most_common(3)}, unlinked {dict(unlinked)}")
    # ------------------------------------------------------------------------------------

    # Optionally limit to the 8 highest satellites per gateway:
    sat_feeder_links = find_feeder_links(bb_operating_satrec_by_name, sc.gateways, sc.unix_s, max_per_gateway=8)

    ground_links = find_ground_links(sc.gateways, sc.pops)
    
    save_html(sc, why, intra_isl_pairs, a.out, sat_feeder_links=sat_feeder_links, ground_links = ground_links,
          satrec_by_name=satrec_by_name,  bb_operating_satrec_by_name=bb_operating_satrec_by_name,
          version_by_name=version_by_name, min_elev_deg=a.min_elev, frame_step=a.frame_step, frame_ms=a.frame_ms)
    
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
