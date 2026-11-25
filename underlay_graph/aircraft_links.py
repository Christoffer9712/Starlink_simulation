"""
aircraft_links.py

Aircraft-satellite links over time.

At every network update the aircraft selects the highest satellites above
the minimum elevation (at most max_links, e.g. one per aero terminal).
Between updates the links are kept; a link breaks when its satellite drops
below the minimum elevation, and is not replaced until the next update.

Network updates happen at given ground-truth times (update_times) or every
update_interval_s seconds, exactly as for feeder links.

Results use the same pair format as the other link functions, one list per
time step:
    aircraft_links[i] = [[aircraft_name, satellite_name], ...]  at unix_s[i]

Usage:
    aircraft_links = find_aircraft_links(bb_operating_satrec_by_name, sc.sat_names,
                                         sc.sat_elev_from_aircraft, sc.unix_s)
"""
from typing import Optional, Sequence

import numpy as np

from simulation_param import *
from underlay_graph.feeder_links import make_update_index


def find_aircraft_links(satrec_by_name, sat_names, sat_elev_from_aircraft, unix_s,
                        min_elev_deg: float = MIN_AIRCRAFT_ELEV_DEG,
                        max_links: Optional[int] = 1,
                        update_interval_s: float = NETWORK_UPDATE_INTERVAL,
                        update_times: Optional[Sequence[float]] = None,
                        aircraft_name: str = "AIRCRAFT"):
    """Return a list with one entry per time step: [[aircraft, satellite], ...].

    satrec_by_name         -- satellites the aircraft may use (only the keys,
                              i.e. names, are used), e.g. operating broadband
    sat_names              -- names of the columns of sat_elev_from_aircraft
                              (sc.sat_names)
    sat_elev_from_aircraft -- (time steps, satellites) elevation in degrees
    unix_s                 -- times (seconds since 1970-01-01 UTC), e.g. sc.unix_s
    min_elev_deg           -- minimum elevation of a usable satellite
    max_links              -- satellites per update (1 per aero terminal);
                              None keeps all usable satellites
    update_interval_s      -- interval between updates when update_times is None
    update_times           -- ground-truth update times (Unix seconds); the
                              aircraft reselects at the first time step at or
                              after each of them. The first step always selects.
    aircraft_name          -- node name of the aircraft in the pairs
    """
    unix_s = np.atleast_1d(np.asarray(unix_s, dtype=float))
    elev = np.asarray(sat_elev_from_aircraft, dtype=float)
    names = np.asarray(sat_names)
    if elev.shape != (len(unix_s), len(names)):
        raise ValueError(f"sat_elev_from_aircraft has shape {elev.shape}, expected "
                         f"({len(unix_s)} time steps, {len(names)} satellites)")

    candidate = np.isin(names, list(satrec_by_name))
    update_index = make_update_index(unix_s, update_interval_s, update_times)

    links_per_step = []
    current = np.array([], dtype=int)                    # indices of linked satellites
    last_update = None
    for i, t in enumerate(unix_s):
        e = elev[i]
        update = update_index(t)

        if update != last_update:
            # Network update: pick the highest usable satellites
            usable = np.where(candidate & (e >= min_elev_deg))[0]
            current = usable[np.argsort(-e[usable])][:max_links]
            last_update = update
        else:
            # Between updates: keep the links, drop satellites that have set
            current = current[e[current] >= min_elev_deg]

        links_per_step.append(sorted([[aircraft_name, names[k]] for k in current]))
    return links_per_step