"""Aircraft trajectory: load the tracker CSV and resample it to a time grid."""
import numpy as np
import pandas as pd

from .geometry import ecef_to_lla, lla_to_ecef

FEET_TO_M = 0.3048


def load_flight(csv_path):
    """Read a Flightradar24-style CSV.

    Expected columns: Timestamp (unix s), UTC, Callsign, Position ("lat,lon"),
    Altitude (ft), Speed (kt), Direction (deg).
    Returns a DataFrame with unix_s, lat, lon, alt_m sorted by time.
    """
    df = pd.read_csv(csv_path)
    latlon = df["Position"].str.split(",", expand=True).astype(float)
    out = pd.DataFrame({
        "unix_s": df["Timestamp"].astype(np.int64),
        "lat": latlon[0],
        "lon": latlon[1],
        "alt_m": df["Altitude"].astype(float) * FEET_TO_M,
    })
    return (out.drop_duplicates("unix_s")
               .sort_values("unix_s")
               .reset_index(drop=True))


def resample_flight(flight, unix_s):
    """Interpolate the track onto the requested unix timestamps.

    Horizontal position is interpolated in Earth-fixed xyz (so it behaves
    across the antimeridian and near the poles); altitude linearly.
    Times outside the recorded track are clamped to the first/last sample.

    Returns (lat, lon, alt_m) arrays, each shaped like `unix_s`.
    """
    surface = lla_to_ecef(flight["lat"].to_numpy(), flight["lon"].to_numpy())
    t_src = flight["unix_s"].to_numpy(float)

    xyz = np.stack([np.interp(unix_s, t_src, surface[:, k]) for k in range(3)], axis=-1)
    lat, lon, _ = ecef_to_lla(xyz)
    alt = np.interp(unix_s, t_src, flight["alt_m"].to_numpy())
    return lat, lon, alt
