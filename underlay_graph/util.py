from simulation_param import EARTH_RADIUS_KM
from sgp4.api import jday

"""
Starlink satellite version from the launch (TLE international designator).

Source: launch table in Wikipedia's "List of Starlink and Starshield
launches" (version per launch, checked against data up to Oct 2026).
"""

# v1.5 launches that flew after the first v2 Mini launch (2023-026)
V15_AFTER_FIRST_V2_MINI = {
    "23028", "23037", "23042", "23046", "23058", "23061", "23064",
    "23065", "23078", "23083", "23088", "23090", "23094", "23099",
}

# Starship launches carrying v3 satellites into orbit
V3_LAUNCHES = {"26225"}            # Group 31-1, 28 Sep 2026

FIRST_SIX_DIGIT_SATNUM = 100000   # catalogued from 11 July 2026 onward

def satversion(satrec):
    """Return 'v0.9', 'v1.0', 'v1.5', 'v2 mini', 'v3' or 'unknown'."""
    designator = satrec.intldesg.strip()          # e.g. '23026A'
    if len(designator) < 5:
        # No designator (common for Alpha-5 TLEs): everything catalogued after
        # 11 July 2026 and before the first v3 launch (28 Sep 2026) is a v2 mini
        return "v2 mini" if satrec.satnum >= FIRST_SIX_DIGIT_SATNUM else "unknown"
    
    designator = satrec.intldesg.strip()          # e.g. '23026A'
    if len(designator) < 5:
        return "unknown"
    launch_id = designator[:5]                    # e.g. '23026'
    year, number = 2000 + int(designator[:2]), int(designator[2:5])
    launch = (year, number)

    if launch < (2019, 74):
        return "v0.9"
    if launch <= (2021, 44):
        return "v1.0"                             # incl. Transporter-1 (2021-006)
    if launch < (2023, 26):
        return "v1.5"
    if launch_id in V15_AFTER_FIRST_V2_MINI:
        return "v1.5"
    if launch_id in V3_LAUNCHES:
        return "v3"
    return "v2 mini"

def mean_altitude_km(satrec):
    """Mean altitude from the TLE's semi-major axis (satrec.a is in Earth radii)."""
    return (satrec.a - 1.0) * EARTH_RADIUS_KM

def tle_age_days(satrec, jd, fr):
    return abs((jd - satrec.jdsatepoch) + (fr - satrec.jdsatepochF))


def newest_epoch(satrecs):
    s = max(satrecs, key=lambda s: s.jdsatepoch + s.jdsatepochF)
    return s.jdsatepoch, s.jdsatepochF


def to_julian(when):
    return jday(when.year, when.month, when.day, when.hour, when.minute,
                when.second + when.microsecond / 1e6)