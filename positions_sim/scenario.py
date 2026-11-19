"""A Scenario holds every node position on a common 1-second time grid.

    scenario = build_scenario(flight_csv, tle_file, gateway_kml, pop_json,
                              start_unix=..., duration_s=900)
    snap = scenario.snapshot(42)     # everything at the 43rd second

`Snapshot` is the single hand-off point for later steps (e.g. building a
NetworkX graph per time step): it carries positions only, no plotting logic.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from skyfield.api import load

from .flight import load_flight, resample_flight
from .geometry import ecef_to_lla, elevation_deg, lla_to_ecef, up_vector
from .ground import load_gateways, load_pops
from .satellites import load_tles, propagate_ecef, select_closest_epoch


@dataclass
class Snapshot:
    """All node positions at one instant. Positions are Earth-fixed metres."""
    unix_s: int
    aircraft_xyz: np.ndarray        # (3,)
    aircraft_lla: np.ndarray        # (3,) lat, lon, alt_m
    sat_names: list                 # N
    satrec_by_name: dict            # N
    sat_xyz: np.ndarray             # (N, 3)
    sat_elev_from_aircraft: np.ndarray  # (N,) degrees
    gateways: pd.DataFrame          # static
    pops: pd.DataFrame              # static

    @property
    def utc(self):
        return pd.Timestamp(self.unix_s, unit="s", tz="UTC")


@dataclass
class Scenario:
    unix_s: np.ndarray              # (T,) int, 1 s apart
    aircraft_lla: np.ndarray        # (T, 3)
    aircraft_xyz: np.ndarray        # (T, 3)
    sat_names: list                 # N
    satrec_by_name:dict             # N
    sat_xyz: np.ndarray             # (T, N, 3) float32
    sat_elev_from_aircraft: np.ndarray  # (T, N) float32, degrees
    flight_track: pd.DataFrame      # raw track (for drawing the full path)
    gateways: pd.DataFrame
    pops: pd.DataFrame

    def __len__(self):
        return len(self.unix_s)

    def snapshot(self, i):
        return Snapshot(
            unix_s=int(self.unix_s[i]),
            aircraft_lla=self.aircraft_lla[i],
            aircraft_xyz=self.aircraft_xyz[i],
            sat_names=self.sat_names,
            satrec_by_name = self.satrec_by_name,
            sat_xyz=self.sat_xyz[i],
            sat_elev_from_aircraft=self.sat_elev_from_aircraft[i],
            gateways=self.gateways,
            pops=self.pops,
        )

    def sat_lla(self):
        """(lat, lon, alt) arrays, each (T, N) — sub-satellite points."""
        return ecef_to_lla(self.sat_xyz)


def build_scenario(flight_csv, tle_file, gateway_kml, pop_json,
                   start_unix=None, duration_s=900, max_tle_age_days=3.0):
    """Load all inputs and compute positions every second in
    [start_unix, start_unix + duration_s). `start_unix` defaults to the first
    flight sample; the window is clipped to the end of the flight.
    """
    flight = load_flight(flight_csv)
    start = int(flight["unix_s"].iloc[0] if start_unix is None else start_unix)
    end = min(start + duration_s, int(flight["unix_s"].iloc[-1]) + 1)
    unix_s = np.arange(start, end, dtype=np.int64)

    # Aircraft
    lat, lon, alt = resample_flight(flight, unix_s)
    aircraft_lla = np.stack([lat, lon, alt], axis=-1)
    aircraft_xyz = lla_to_ecef(lat, lon, alt)

    # Satellites
    tles = select_closest_epoch(load_tles(tle_file), unix_s[len(unix_s) // 2],
                                max_age_days=max_tle_age_days)
    sat_xyz = propagate_ecef(tles, unix_s, load.timescale())
    elev = elevation_deg(aircraft_xyz[:, None, :], up_vector(lat, lon)[:, None, :],
                         sat_xyz).astype(np.float32)

    return Scenario(
        unix_s=unix_s,
        aircraft_lla=aircraft_lla,
        aircraft_xyz=aircraft_xyz,
        sat_names=[t.name for t in tles],
        satrec_by_name = {t.name: t.satrec for t in tles},
        sat_xyz=sat_xyz,
        sat_elev_from_aircraft=elev,
        flight_track=flight,
        gateways=load_gateways(gateway_kml),
        pops=load_pops(pop_json),
    )
