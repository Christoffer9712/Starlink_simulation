"""
operational.py

Flag Starlink satellites that are probably not in service, from TLE data alone,
and describe the operational constellation as shells and orbital planes.

Criteria
  1. TLE epoch far from the analysis time (stale or decayed)
  2. Not near-circular (manoeuvring or uncontrolled)
  3. Below 400 km and either young (still raising) or not a direct-to-cell
     version (retiring); only direct-to-cell v2 minis operate that low
  4. Not in a populated shell (raising, lowering, drifting)
  5. SGP4 fails at the analysis time
  6. Newer satellite sharing a slot with an older one
     (close together AND moving together)

Pass the whole constellation (one TLE snapshot), not a subset such as only
the satellites visible from the aircraft: shells and slots are found by
comparing satellites with each other.

Usage:
    satrec_by_name = {t.name: t.satrec for t in tles}
    dead = find_non_operational(sat_names, satrec_by_name)
    why = non_operational_reasons(sat_names, satrec_by_name)   # for checking

    shells = find_shells(list(satrec_by_name), satrec_by_name, version_by_name)
    plane_by_name = find_planes(shells, satrec_by_name, when)  # name -> (shell_id, plane_id)
    print_shells(shells)
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from underlay_graph.util import satversion, mean_altitude_km, newest_epoch, tle_age_days, to_julian
from simulation_param import *


def find_non_operational(sat_names, satrec_by_name, when: Optional[datetime] = None):
    """Return a sorted list of satellite names that are probably not in service."""
    return sorted(non_operational_reasons(sat_names, satrec_by_name, when))


def non_operational_reasons(sat_names, satrec_by_name, when: Optional[datetime] = None):
    """Return {name: reason} for satellites that are probably not in service."""
    satrecs = {n: satrec_by_name[n] for n in set(sat_names) if n in satrec_by_name}
    if not satrecs:
        return {}

    jd, fr = to_julian(when) if when else newest_epoch(satrecs.values())
    reasons = {}

    # 1. Stale TLE
    for name, s in satrecs.items():
        age = tle_age_days(s, jd, fr)
        if age > MAX_TLE_AGE_DAYS:
            reasons[name] = f"TLE epoch {age:.1f} days from analysis time"

    # 2. Not near-circular
    for name, s in satrecs.items():
        if name not in reasons and s.ecco > MAX_ECCENTRICITY:
            reasons[name] = f"eccentric orbit (e = {s.ecco:.4f})"

    # 3. Low satellites: only older direct-to-cell satellites operate below 400 km
    for name, s in satrecs.items():
        if name in reasons:
            continue
        if mean_altitude_km(s) >= LOW_ALTITUDE_KM:
            continue
        version = satversion(s)
        if version not in DIRECT_TO_CELL_VERSIONS and version != "unknown":
            reasons[name] = f"Retiring: {version} below {LOW_ALTITUDE_KM:.0f} km"
        elif s.revnum < YOUNG_REVS:
            reasons[name] = f"Raising: {version} below {LOW_ALTITUDE_KM:.0f} km"

    # 4. Not in a populated shell
    remaining = {n: s for n, s in satrecs.items() if n not in reasons}
    flagged, _ = off_shell(remaining)
    reasons.update(flagged)

    # 5. Propagate the rest to the analysis time
    names, positions, velocities = [], [], []
    for name, s in satrecs.items():
        if name in reasons:
            continue
        error, r, v = s.sgp4(jd, fr)
        if error:
            reasons[name] = "SGP4 error at analysis time"
        else:
            names.append(name)
            positions.append(r)
            velocities.append(v)

    # 6. Shared slots
    for name in newer_in_shared_slot(names, np.array(positions),
                                     np.array(velocities), satrecs):
        reasons[name] = "shares a slot with an older satellite"

    return reasons


@dataclass
class Plane:
    raan_deg: float                 # mean RAAN of the plane at the analysis time
    names: list                     # ordered along the orbit (argument of latitude)


@dataclass
class Shell:
    inclination_deg: float          # median inclination
    altitude_km: float              # median mean altitude (a - equatorial radius)
    altitude_min_km: float
    altitude_max_km: float
    n_satellites: int
    names: list = field(repr=False)
    versions: dict = field(default_factory=dict)   # {version: number of satellites}
    planes: list = field(default_factory=list, repr=False)   # list of Plane


def off_shell(satrecs):
    """Find the populated shells and the satellites outside them.

    Returns (flagged, shells):
      flagged -- {name: reason} for satellites with too few peers at the same
                 inclination and mean altitude
      shells  -- list of Shell, sorted by inclination then altitude
    """
    names = list(satrecs)
    inc = np.degrees([satrecs[n].inclo for n in names])
    alt = np.array([mean_altitude_km(satrecs[n]) for n in names])

    flagged = {}
    in_shell = []
    for k, name in enumerate(names):
        peers = ((np.abs(inc - inc[k]) <= SHELL_INCLINATION_TOL_DEG)
                 & (np.abs(alt - alt[k]) <= SHELL_ALTITUDE_TOL_KM))
        if peers.sum() - 1 < SHELL_MIN_SATELLITES:
            flagged[name] = (f"not in a populated shell "
                             f"(mean alt {alt[k]:.0f} km, incl {inc[k]:.2f}°)")
        else:
            in_shell.append(k)

    shells = []
    for by_inc in split_on_gaps(np.array(in_shell), inc, SHELL_SPLIT_INCLINATION_GAP_DEG):
        for members in split_on_gaps(by_inc, alt, SHELL_SPLIT_ALTITUDE_GAP_KM):
            shells.append(Shell(
                inclination_deg=float(np.median(inc[members])),
                altitude_km=float(np.median(alt[members])),
                altitude_min_km=float(alt[members].min()),
                altitude_max_km=float(alt[members].max()),
                n_satellites=len(members),
                names=[names[k] for k in members],
            ))
    shells.sort(key=lambda s: (s.inclination_deg, s.altitude_km))
    return flagged, shells


def split_on_gaps(idx, values, max_gap):
    """Sort idx by values and split wherever neighbouring values differ by > max_gap."""
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(values[idx])]
    breaks = np.where(np.diff(values[idx]) > max_gap)[0] + 1
    return np.split(idx, breaks)


def find_shells(sat_names, satrec_by_name, version_by_name: Optional[dict] = None):
    """The populated shells of the given satellites (whole constellation recommended).

    version_by_name -- optional {name: version}; if given, each shell's
                       `versions` holds the number of satellites per version.
    """
    satrecs = {n: satrec_by_name[n] for n in set(sat_names) if n in satrec_by_name}
    _, shells = off_shell(satrecs)

    if version_by_name is not None:
        for shell in shells:
            shell.versions = dict(Counter(version_by_name.get(n, "unknown")
                                          for n in shell.names))
    return shells


def find_planes(shells, satrec_by_name, when: Optional[datetime] = None):
    """Split every shell into orbital planes.

    All satellites are propagated to one time (default: newest TLE epoch),
    because the planes drift by several degrees per day. Within a shell,
    satellites are first grouped by RAAN (split wherever the gap exceeds
    PLANE_SPLIT_RAAN_GAP_DEG, also across 360°/0°). Each RAAN group is then
    split by inclination and mean altitude, so interleaved planes from
    neighbouring shells are separated.

    Fills shell.planes and returns {name: (shell_id, plane_id)}, where
    shell_id indexes `shells` and plane_id indexes shells[shell_id].planes.
    Each plane's names are ordered along the orbit.
    """
    names = [n for shell in shells for n in shell.names]
    if not names:
        return {}
    jd, fr = (to_julian(when) if when
              else newest_epoch(satrec_by_name[n] for n in names))

    raan, arg_lat = {}, {}
    for name in names:
        geometry = plane_geometry(satrec_by_name[name], jd, fr)
        if geometry is not None:
            raan[name], arg_lat[name] = geometry

    plane_by_name = {}
    for shell_id, shell in enumerate(shells):
        members = np.array([n for n in shell.names if n in raan])
        angles = np.array([raan[n] for n in members])
        inc = np.degrees([satrec_by_name[n].inclo for n in members])
        alt = np.array([mean_altitude_km(satrec_by_name[n]) for n in members])

        shell.planes = []
        for raan_group in split_circular(angles, PLANE_SPLIT_RAAN_GAP_DEG):
            for by_inc in split_on_gaps(raan_group, inc, PLANE_SPLIT_INCLINATION_GAP_DEG):
                for idx in split_on_gaps(by_inc, alt, PLANE_SPLIT_ALTITUDE_GAP_KM):
                    plane_names = sorted(members[idx], key=lambda n: arg_lat[n])
                    plane_id = len(shell.planes)
                    shell.planes.append(Plane(raan_deg=circular_mean_deg(angles[idx]),
                                              names=plane_names))
                    for name in plane_names:
                        plane_by_name[name] = (shell_id, plane_id)
    return plane_by_name


def plane_geometry(satrec, jd, fr):
    """(RAAN, argument of latitude) in degrees at the given time, or None on error.

    Computed in SGP4's TEME frame. Using ECEF would shift every RAAN by the
    same Earth-rotation angle, so planes and ordering are identical.
    """
    error, r, v = satrec.sgp4(jd, fr)
    if error:
        return None
    r, v = np.array(r), np.array(v)
    h = np.cross(r, v)
    h /= np.linalg.norm(h)                        # orbit normal
    raan = np.arctan2(h[0], -h[1])
    node = np.array([np.cos(raan), np.sin(raan), 0.0])
    ahead = np.cross(h, node)                     # 90 deg past the node
    arg_lat = np.arctan2(r @ ahead, r @ node)
    return float(np.degrees(raan) % 360), float(np.degrees(arg_lat) % 360)


def split_circular(angles, max_gap):
    """Group angles (deg) wherever the gap to the next one exceeds max_gap.

    Works across 360°/0° by starting the sweep after the largest gap.
    Returns a list of index arrays into `angles`.
    """
    if len(angles) == 0:
        return []
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + 360.0))  # last gap wraps
    start = (np.argmax(gaps) + 1) % len(order)
    order = np.roll(order, -start)
    unwrapped = (angles[order] - angles[order[0]]) % 360.0
    breaks = np.where(np.diff(unwrapped) > max_gap)[0] + 1
    return np.split(order, breaks)


def circular_mean_deg(angles):
    """Mean of angles in degrees, correct across 360°/0°."""
    r = np.radians(angles)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def print_shells(shells):
    print(f"{'incl °':>7}  {'alt km':>7}  {'alt range km':>13}  {'sats':>5}  "
          f"{'planes':>6}  versions")
    for s in shells:
        print(f"{s.inclination_deg:7.2f}  {s.altitude_km:7.0f}  "
              f"{s.altitude_min_km:6.0f}–{s.altitude_max_km:<6.0f}  {s.n_satellites:5d}  "
              f"{len(s.planes):6d}  {format_versions(s.versions)}")


def format_versions(versions):
    """'v1.5: 140, v2 mini: 3092' in a fixed version order."""
    def order(version):
        return VERSION_ORDER.index(version) if version in VERSION_ORDER else len(VERSION_ORDER)
    return ", ".join(f"{v}: {n}" for v, n in sorted(versions.items(), key=lambda x: order(x[0])))


def newer_in_shared_slot(names, positions, velocities, satrecs):
    """Newer satellite of every pair that is close together and moving together.

    Positions and velocities are SGP4's TEME output. Only differences between
    satellites are used, which are the same in TEME and ECEF to well below
    the thresholds.
    """
    if len(names) < 2:
        return set()
    newer = set()
    for i, j in cKDTree(positions).query_pairs(SLOT_DISTANCE_KM):
        if np.linalg.norm(velocities[i] - velocities[j]) < SLOT_REL_SPEED_KM_S:
            newer.add(max(names[i], names[j], key=lambda n: launch_key(satrecs[n])))
    return newer


def launch_key(satrec):
    """Sortable launch order: catalog numbers increase with time."""
    return satrec.satnum