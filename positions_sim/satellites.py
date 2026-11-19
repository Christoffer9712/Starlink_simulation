"""Starlink TLEs -> Earth-fixed satellite positions on a time grid.

Propagation uses sgp4's vectorised SatrecArray (all satellites x all times in
one C call) and Skyfield's frame rotations to go from TEME to Earth-fixed
ITRS. This gives the same result as `EarthSatellite.at(t).frame_xyz(itrs)`
but is fast enough for thousands of satellites at 1 s steps.
"""
from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec, SatrecArray
from skyfield.framelib import itrs
from skyfield.sgp4lib import TEME

UNIX_EPOCH_JD = 2440587.5


@dataclass
class Tle:
    name: str
    satnum: int
    satrec: Satrec

    @property
    def epoch_jd(self):
        return self.satrec.jdsatepoch + self.satrec.jdsatepochF


def load_tles(path):
    """Parse a 3-line-element file (name / line 1 / line 2, repeated).

    Files with several epochs per satellite are fine; pick one per satellite
    with `select_closest_epoch`.
    """
    lines = [ln.rstrip() for ln in open(path) if ln.strip()]
    tles = []
    for name, l1, l2 in zip(lines[0::3], lines[1::3], lines[2::3]):
        sat = Satrec.twoline2rv(l1, l2)
        tles.append(Tle(name.removeprefix("0 ").strip(), sat.satnum, sat))
    return tles


def select_closest_epoch(tles, unix_s, max_age_days=3.0):
    """Keep, per satellite number, the TLE whose epoch is closest to `unix_s`.

    Satellites whose best TLE is more than `max_age_days` away are dropped
    (decayed, not yet launched, or simply stale). Returned sorted by satnum.
    """
    jd = UNIX_EPOCH_JD + unix_s / 86400.0
    best = {}
    for tle in tles:
        cur = best.get(tle.satnum)
        if cur is None or abs(tle.epoch_jd - jd) < abs(cur.epoch_jd - jd):
            best[tle.satnum] = tle
    keep = [t for t in best.values() if abs(t.epoch_jd - jd) <= max_age_days]
    return sorted(keep, key=lambda t: t.satnum)


def propagate_ecef(tles, unix_s, ts):
    """Earth-fixed positions in metres, shape (T, N, 3), float32.

    `unix_s` is a 1-D array of integer unix timestamps (UTC); `ts` is a
    Skyfield timescale. Entries where SGP4 fails are NaN.
    """
    unix_s = np.asarray(unix_s, dtype=np.int64)
    days, secs = np.divmod(unix_s, 86400)
    jd, fr = UNIX_EPOCH_JD + days.astype(float), secs / 86400.0

    err, r_teme_km, _ = SatrecArray([t.satrec for t in tles]).sgp4(jd, fr)  # (N, T, 3)
    r_teme_km[err != 0] = np.nan

    # TEME -> GCRS -> ITRS rotation for every time step: (3, 3, T).
    # (Day + seconds-of-day keeps Skyfield from treating unix seconds as
    #  elapsed SI seconds, which would be off by the accumulated leap seconds.)
    t = ts.utc(1970, 1, 1 + days, 0, 0, secs.astype(float))
    rot = np.einsum("ijt,kjt->ikt", itrs.rotation_at(t), TEME.rotation_at(t))

    r_itrs_km = np.einsum("ijt,ntj->tni", rot, r_teme_km)                   # (T, N, 3)
    return (r_itrs_km * 1000.0).astype(np.float32)
