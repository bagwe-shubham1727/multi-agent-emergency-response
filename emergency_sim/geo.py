"""Schematic city map and travel-time model.

The city is a 20 km x 20 km grid. Vehicles move along streets, so distance is
Manhattan distance. Traffic is slower near downtown (10, 10): the congestion
factor rises from 1.0 at the edge of downtown to 1.5 at its centre. The data
generator and the simulation engine both import this module, so the travel
times in travel_times.csv are the same numbers the engine will use.
"""

MAP_SIZE_KM = 20.0
DOWNTOWN = (10.0, 10.0)
DOWNTOWN_RADIUS_KM = 6.0
MAX_CONGESTION_PENALTY = 0.5

# km per minute with lights and sirens
SPEED_KM_PER_MIN = {"ambulance": 0.9, "responder": 0.8}
# minutes from alert to wheels rolling
TURNOUT_MIN = {"ambulance": 1.0, "responder": 2.0}

AVENUES = ["1st Ave", "2nd Ave", "3rd Ave", "4th Ave", "5th Ave",
           "6th Ave", "7th Ave", "8th Ave", "9th Ave", "10th Ave"]
STREETS = ["Ash St", "Birch St", "Cedar St", "Dogwood St", "Elm St",
           "Fir St", "Grove St", "Hazel St", "Ivy St", "Juniper St"]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def congestion(p):
    d = ((p[0] - DOWNTOWN[0]) ** 2 + (p[1] - DOWNTOWN[1]) ** 2) ** 0.5
    return 1.0 + MAX_CONGESTION_PENALTY * max(0.0, 1.0 - d / DOWNTOWN_RADIUS_KM)


def travel_time(a, b, vehicle="ambulance", include_turnout=False):
    """Minutes to drive from a to b (each an (x, y) tuple in km)."""
    factor = (congestion(a) + congestion(b)) / 2
    minutes = manhattan(a, b) / SPEED_KM_PER_MIN[vehicle] * factor
    if include_turnout:
        minutes += TURNOUT_MIN[vehicle]
    return round(minutes, 1)


def nearest_intersection(p):
    """Human-readable cross streets for a point, e.g. 'Elm St & 4th Ave'."""
    ave = AVENUES[min(int(p[0] / 2), len(AVENUES) - 1)]
    st = STREETS[min(int(p[1] / 2), len(STREETS) - 1)]
    return f"{st} & {ave}"
