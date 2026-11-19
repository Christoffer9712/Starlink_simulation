"""Static ground infrastructure: Starlink gateways (KML) and PoPs (JSON)."""
import json
import re
import xml.etree.ElementTree as ET

import pandas as pd

from .geometry import lla_to_ecef


def _snake(name):
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def _add_ecef(df):
    xyz = lla_to_ecef(df["lat"].to_numpy(), df["lon"].to_numpy(),
                      df["alt"].to_numpy())
    df["x"], df["y"], df["z"] = xyz.T
    return df


def load_gateways(kml_path):
    """One row per <Placemark> with a <Point>.

    Columns: name, lat, lon, alt [m], x, y, z [m, Earth-fixed], plus every
    <ExtendedData> field in snake_case (status, ka_antenna_count,
    ka_min_elevation, ...).
    """
    rows = []
    for pm in ET.parse(kml_path).getroot().findall(".//{*}Placemark"): 
        coords = pm.findtext(".//{*}Point/{*}coordinates")
        if not coords:
            continue
        # I added XX to GWs not known to be operational
        if pm.findtext("{*}name") and pm.findtext("{*}name").endswith('XX'):
            continue

        lon, lat, *alt = (float(v) for v in coords.strip().split(","))
        row = {"name": (pm.findtext("{*}name") or "").strip(),
               "lat": lat, "lon": lon, "alt": alt[0] if alt else 0.0}
        for data in pm.findall(".//{*}Data"):
            row[_snake(data.get("name", ""))] = (data.findtext("{*}value") or "").strip()
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("ka_antenna_count", "ka_antenna_size", "ka_min_elevation",
                "e_antenna_count", "e_antenna_size", "e_min_elevation"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _add_ecef(df)


def load_pops(json_path):
    """One row per PoP. Columns: code, city, country, lat, lon, alt, x, y, z, ..."""
    with open(json_path) as f:
        df = pd.DataFrame(json.load(f))
    df["alt"] = 0.0
    return _add_ecef(df)
