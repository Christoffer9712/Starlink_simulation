from skyfield.api import load, EarthSatellite, wgs84
from skyfield.framelib import itrs
import numpy as np

ts = load.timescale()
L = open("starlink_3le.txt").read().strip().splitlines()
sats = [EarthSatellite(L[i+1], L[i+2], L[i].lstrip("0 "), ts) for i in range(0, len(L), 3)]

# group by satellite number; keep the epoch closest to the time you need
by_id = {}
for s in sats:
    by_id.setdefault(s.model.satnum, []).append(s)

def tle_at(satnum, t):
    return min(by_id[satnum], key=lambda s: abs(s.epoch.tt - t.tt))

t = ts.utc(2026, 8, 21, 0, 47, np.arange(0, 900, 15))   # every 15 s for 15 min
sat = tle_at(next(iter(by_id)), t[0])
xyz_km = sat.at(t).frame_xyz(itrs).km                    # Earth-fixed coordinates