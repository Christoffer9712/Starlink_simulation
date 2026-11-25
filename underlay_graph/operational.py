from simulation_param import MAX_ECCENTRICITY, SLOT_DISTANCE_KM, SLOT_REL_SPEED_KM_S, SHELL_INCLINATION_TOL_DEG
import numpy as np
from underlay_graph.util import to_julian
from scipy.spatial import cKDTree

def stable_satrec(now, prevs, max_change_km):
    def is_stable(name, s):
        return all(name in p and abs(s.a - p[name].a) * s.radiusearthkm < max_change_km
                   for p in prevs)
    stable = {n: s for n, s in now.items() if is_stable(n, s)}
    unstable = {n: s for n, s in now.items() if n not in stable}
    return stable, unstable

def non_operational_reasons(satrec_by_name, when):
    jd, fr = to_julian(when)
    
    reasons = {}
    
    # Not near-circular
    for name, s in satrec_by_name.items():
        if s.ecco > MAX_ECCENTRICITY:
            reasons[name] = f"eccentric orbit (e = {s.ecco:.4f})"

    # Propagate the rest to the analysis time
    names, positions, velocities = [], [], []
    for name, s in satrec_by_name.items():
        if name in reasons:
            continue
        error, r, v = s.sgp4(jd, fr)
        if error:
            reasons[name] = "SGP4 error at analysis time"
        else:
            names.append(name)
            positions.append(r)
            velocities.append(v)

    # Shared slots
    for name in newer_in_shared_slot(names, np.array(positions),
                                        np.array(velocities), satrec_by_name):
        reasons[name] = "shares a slot with an older satellite"

    return reasons

# ---- shared slots --------------------------------------------------------------

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