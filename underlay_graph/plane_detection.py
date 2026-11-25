from underlay_graph.util import to_julian
import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class Plane:
    index: int                  # 0, 1, 2, ... in order of RAAN
    raan_deg: float             # median RAAN at `when`
    raan_min_deg: float
    raan_max_deg: float
    n_satellites: int
    sat_names: list = field(default_factory=list)


def find_planes(satrec_by_name, when, eps_deg, min_samples):
    """Group the satellites of one shell into orbital planes by their RAAN at `when`.

    Returns (planes, plane_by_name, unassigned).
    """
    jd, fr = to_julian(when)

    # 1. RAAN of every satellite at `when` (TLE epochs differ and RAAN drifts)
    names, raan, unassigned = [], [], []
    for name, s in satrec_by_name.items():
        err, r, v = s.sgp4(jd, fr)
        if err:
            unassigned.append(name)
            continue
        hx, hy, _ = np.cross(r, v)                       # orbit normal
        names.append(name)
        raan.append(np.degrees(np.arctan2(hx, -hy)) % 360)
    names, raan = np.array(names), np.array(raan)

    # 2. Cluster on the unit circle so that 359° and 1° are neighbours
    xy = np.column_stack([np.cos(np.radians(raan)), np.sin(np.radians(raan))])
    labels = DBSCAN(eps=np.radians(eps_deg), min_samples=min_samples).fit_predict(xy)

    # 3. One Plane per cluster, numbered in order of RAAN
    planes = [make_plane(names[labels == k], raan[labels == k]) for k in set(labels) - {-1}]
    planes.sort(key=lambda p: p.raan_deg)
    for i, p in enumerate(planes):
        p.index = i

    plane_by_name = {n: p.index for p in planes for n in p.sat_names}
    unassigned += names[labels == -1].tolist()
    return planes, plane_by_name, unassigned


def make_plane(names, raan):
    """RAAN stats measured as offsets from the first member, so a plane across 0° doesn't break."""
    ref = raan[0]
    offset = (raan - ref + 180) % 360 - 180              # -180..180 around ref
    return Plane(
        index=-1,
        raan_deg=float((ref + np.median(offset)) % 360),
        raan_min_deg=float((ref + offset.min()) % 360),
        raan_max_deg=float((ref + offset.max()) % 360),
        n_satellites=len(names),
        sat_names=names.tolist(),
    )

def plot_plane_raan(shell, filename):
    """Satellites per plane and RAAN gap to the next plane, for checking the plane split."""
    raan = np.array([p.raan_deg for p in shell.planes])
    size = [p.n_satellites for p in shell.planes]
    gap = np.diff(np.append(raan, raan[0] + 360))        # to the next plane, wrapping at 360°
    typical = np.median(gap)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    top.bar(raan, size, width=1.0)
    top.set_ylabel("Satellites per plane")

    bottom.bar(raan, gap, width=1.0)
    bottom.axhline(typical, color="black", linestyle="--", linewidth=0.8,
                   label=f"median gap {typical:.2f}°")
    bottom.set_ylabel("Gap to next plane (°)")
    bottom.set_xlabel("RAAN (°)")
    bottom.legend(loc="upper right")

    fig.suptitle(f"{shell.label}: {len(raan)} planes")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)