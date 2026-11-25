from dataclasses import dataclass, field

"""Known Starlink shells (mean altitude = semi-major axis - equatorial radius)."""

@dataclass
class ShellDefinition:
    """A known shell: satellites within the tolerances (and of an allowed
    version, if versions is given) belong to it. Altitude is mean altitude,
    i.e. semi-major axis minus the equatorial radius, as in print_shells."""
    label: str
    inclination_deg: float
    altitude_km: float
    inclination_tol_deg: float = 0.05
    altitude_tol_km: float = 1.0
    versions: list = field(default_factory=list) 

BB_SHELL_DEFINITIONS = [
    ShellDefinition("43° / 485 km",     43.00, 485.0, altitude_tol_km=3.0),
    ShellDefinition("53.16° / 460 km",  53.16, 460.0),
    ShellDefinition("53.16° / 463 km",  53.16, 463.0),
    ShellDefinition("53.16° / 465.5 km", 53.16, 465.5),
    ShellDefinition("53.16° / 471 km",  53.16, 471.0),
    ShellDefinition("53.16° / 475.5 km", 53.16, 475.5, versions=["v2 mini"]),
    ShellDefinition("53.16° / 540 km",  53.16, 540.0, altitude_tol_km=3.0,
                    versions=["v1.5"]),
    ShellDefinition("70° / 570 km",     70.00, 570.0, altitude_tol_km=3.0),
    ShellDefinition("97.28° / 546 km",  97.28, 546.0, altitude_tol_km=3.0),
    ShellDefinition("97.28° / 462 km",  97.28, 462.0, altitude_tol_km=3.0),
]

DTC_SHELL_DEFINITIONS = [
    ShellDefinition("DTC 43° / 358 km",    43.00, 358.0, altitude_tol_km=2.0,
                    versions=["v2 mini"]),
    ShellDefinition("DTC 53.16° / 360 km", 53.16, 360.0, altitude_tol_km=2.0,
                    versions=["v2 mini"]),
]