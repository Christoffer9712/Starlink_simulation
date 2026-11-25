from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from simulation_param import VERSION_ORDER
from collections import Counter
import matplotlib.pyplot as plt

@dataclass
class Shell:
    inclination_deg: float          # median inclination
    altitude_km: float              # median mean altitude (a - equatorial radius)
    altitude_min_km: float
    altitude_max_km: float
    n_satellites: int
    planes: list = field(default_factory=list)   # list of Plane
    versions: dict = field(default_factory=dict)   # dict of versions:number_of_sats in shell
    label: str = ""                 # e.g. "53.16° / 463 km"


def find_shells_dbscan(satrec_by_name, version_by_name, eps, min_samples, alt_scale, inc_scale):
    
    sats_recs = list(satrec_by_name.values())
    sat_names = np.array(list(satrec_by_name.keys()))

    # Orbital elements per satellite
    inc = np.degrees([s.inclo for s in sats_recs])                         # deg
    alt = np.array([(s.a - 1.0) * s.radiusearthkm for s in sats_recs])     # km (s.a is in Earth radii)

    # Scale features so eps=1 means "within ~0.5° inclination / ~10 km altitude"
    X = np.column_stack([inc / inc_scale, alt / alt_scale])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    shells, sat_name_to_shell = [], {}
    for k in set(labels) - {-1}:          # -1 = noise (orbit-raising, deorbiting, strays)
        m = labels == k
        i, h = float(np.median(inc[m])), float(np.median(alt[m]))
        sn = sat_names[m]
        versions = Counter([version_by_name[n] for n in sn])
        label = f"{i:.2f}° / {h:.0f} km"
        sat_name_to_shell.update(dict.fromkeys(sn.tolist(), label))

        shells.append(Shell(
            inclination_deg=i,
            altitude_km=h,
            altitude_min_km=float(alt[m].min()),
            altitude_max_km=float(alt[m].max()),
            n_satellites=int(m.sum()),
            versions=versions,
            label=label,
        ))
    noise = sat_names[labels == -1].tolist()

    return sorted(shells, key=lambda s: (s.inclination_deg, s.altitude_km)), noise, sat_name_to_shell

def plot_shells(satrec_by_name, shells):
    sats = list(satrec_by_name.values())
    inc = np.degrees([s.inclo for s in sats])
    alt = np.array([(s.a - 1.0) * s.radiusearthkm for s in sats])

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [np.arange(inc.min() - 1, inc.max() + 1, 0.5),   # 0.5° wide
            np.arange(alt.min() - 5, alt.max() + 5, 2)]     # 2 km tall
    *_, img = ax.hist2d(inc, alt, bins=bins, cmin=1, norm=LogNorm(), cmap="viridis")
    fig.colorbar(img, ax=ax, label="Satellites per bin")

    for sh in shells:
        ax.add_patch(Rectangle((sh.inclination_deg - 0.5, sh.altitude_min_km),
                               1.0, sh.altitude_max_km - sh.altitude_min_km,
                               fill=False, edgecolor="black"))
        ax.text(sh.inclination_deg + 0.6, sh.altitude_km,
                f"{sh.label} ({sh.n_satellites})", va="center", fontsize=8)

    ax.set_xlim(40, 110)   # room for the labels on the right
    ax.set_xlabel("Inclination (°)")
    ax.set_ylabel("Mean altitude (km)")
    fig.tight_layout()
    fig.savefig(f"shells.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def print_shells(shells):
    print(f"{'label':>18}  {'incl °':>7}  {'alt km':>7}  {'alt range km':>13}  {'sats':>5}  "
          f"{'planes':>6}  versions")
    for s in shells:
        print(f"{s.label:>18}  {s.inclination_deg:7.2f}  {s.altitude_km:7.0f}  "
              f"{s.altitude_min_km:6.0f}–{s.altitude_max_km:<6.0f}  {s.n_satellites:5d}  "
              f"{len(s.planes):6d}  {format_versions(s.versions)}")

def format_versions(versions):
    """'v1.5: 140, v2 mini: 3092' in a fixed version order."""
    def order(version):
        return VERSION_ORDER.index(version) if version in VERSION_ORDER else len(VERSION_ORDER)
    return ", ".join(f"{v}: {n}" for v, n in sorted(versions.items(), key=lambda x: order(x[0])))

def plot_shell_distribution(shell, satrec_in_shell, version_by_name, filename):
    """Debug: inclination vs mean altitude of one shell, coloured by version, to spot sub-shells."""
    names = list(satrec_in_shell)
    sats = [satrec_in_shell[n] for n in names]
    inc = np.degrees([s.inclo for s in sats])
    alt = np.array([(s.a - 1.0) * s.radiusearthkm for s in sats])
    ver = np.array([version_by_name.get(n, "unknown") for n in names])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 1]})
    bins = np.arange(alt.min(), alt.max() + 1, 1)       # 1 km altitude bins
    for version, colour in zip(sorted(set(ver)), ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
        m = ver == version
        left.scatter(inc[m], alt[m], s=4, alpha=0.4, color=colour, label=f"{version} ({m.sum()})")
        right.hist(alt[m], bins=bins, orientation="horizontal", histtype="step", color=colour)

    left.set_xlabel("Inclination (°)")
    left.set_ylabel("Mean altitude (km)")
    left.legend(markerscale=4)
    right.set_xlabel("Satellites per km")
    fig.suptitle(f"{shell.label}: {len(names)} satellites")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)