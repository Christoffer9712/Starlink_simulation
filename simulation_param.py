EARTH_RADIUS_KM = 6378.137

MAX_TLE_AGE_DAYS = 7.0
MAX_ECCENTRICITY = 0.002          # operational Starlinks are near-circular

# Below this altitude only direct-to-cell satellites (v2 mini) operate
LOW_ALTITUDE_KM = 400.0
DIRECT_TO_CELL_VERSIONS = {"v2 mini"}
YOUNG_REVS = 1500                 # ~100 days after launch at ~15.5 rev/day

# A shell = at least this many other satellites with similar inclination and altitude
SHELL_MIN_SATELLITES = 60         # above the size of one launch batch
SHELL_INCLINATION_TOL_DEG = 0.2
SHELL_ALTITUDE_TOL_KM = 3.0

# Same slot = closer than this and moving together
SLOT_DISTANCE_KM = 50.0
SLOT_REL_SPEED_KM_S = 0.1         # different planes cross at ~0.5 km/s or more

# Splitting the in-shell satellites into separate shells:
# a gap larger than this between neighbouring values starts a new shell
SHELL_SPLIT_INCLINATION_GAP_DEG = SHELL_INCLINATION_TOL_DEG
SHELL_SPLIT_ALTITUDE_GAP_KM = 2 * SHELL_ALTITUDE_TOL_KM

# Order in which versions are printed
VERSION_ORDER = ["v0.9", "v1.0", "v1.5", "v2 mini", "v3", "unknown"]

# RAAN gap that separates two orbital planes
PLANE_SPLIT_RAAN_GAP_DEG = 0.5

# Longest link accepted (keeps the line of sight well above the atmosphere)
MAX_LINK_RANGE_KM = 4500.0

PLANE_SPLIT_INCLINATION_GAP_DEG = 0.05   # merged planes from 53.0° and 53.2° separate
PLANE_SPLIT_ALTITUDE_GAP_KM = 3.0
MIN_PLANE_SATELLITES = 4                 # smaller "planes" = satellites in transit

NETWORK_UPDATE_INTERVAL = 15.0    # s, gateways reselect their satellites at this interval

MIN_GW_ELEV_DEG = 25.0            # minimum elevation for a gateway-satellite feeder link