"""
Load and parse plane data from a .txt file to a numpy array with additional metrics.
"""
import re
import numpy as np
from typing import List
from pydantic import BaseModel

COLS = {
    "T_APPEAR": 0,
    "T_LAND_EARLY": 1,
    "T_LAND_TARGET": 2,
    "T_LAND_ASSIGNED": 3,
    "T_LAND_LATE": 4,
    "P_LAND_EARLY": 5,
    "P_LAND_LATE": 6
}

class PlaneData(BaseModel):
    t_appear: float
    t_early: float
    t_target: float
    t_assigned: float
    t_late: float
    p_early: float
    p_late: float

class ScheduleData(BaseModel):
    n_planes: int
    t_freeze: float
    data: List[PlaneData]

def split_terms(line):
    """
    Split a line of numbers into a list of floats
    """
    nums = re.split(r"\s+", line.strip())
    return [float(num) for num in nums]

def load_data(path):
    """
    Load and parse plane data from a .txt file to a numpy array with additional metrics and 
    randomised assigned landing time.
    """
    with open(path, 'r', encoding="utf-8") as file:

        schedule = ScheduleData()

        # Get the plane count and freeze time from the first line.
        next_line = file.readline()

        terms = None if not next_line else split_terms(next_line)

        n_planes = int(terms[0])
        t_freeze = terms[1]
        # print(f"Plane count: {n_planes}")
        # print(f"Freeze time: {t_freeze}")

        # Keep reading all remaining terms in groups of 6 + n_planes
        # columns = plane attributes + n_planes separation
        # rows = n_planes
        var_per_plane = len(COLS) + n_planes
        data = np.zeros([n_planes, var_per_plane])

        term_idx = 0
        plane_idx = 0
        while True:
            next_line = file.readline()

            # Exit if end of file
            if not next_line:
                break

            terms = None if not next_line else split_terms(next_line)

            for term in terms:
                if term_idx == 0:
                    data[plane_idx, COLS["T_APPEAR"]] = term
                elif term_idx == 1:
                    data[plane_idx, COLS["T_LAND_EARLY"]] = term
                elif term_idx == 2:
                    data[plane_idx, COLS["T_LAND_TARGET"]] = term
                elif term_idx == 3:
                    data[plane_idx, COLS["T_LAND_LATE"]] = term
                elif term_idx == 4:
                    data[plane_idx, COLS["P_LAND_EARLY"]] = term
                elif term_idx == 5:
                    data[plane_idx, COLS["P_LAND_LATE"]] = term
                else:
                    if term_idx == 6:
                        term_idx += 1

                    data[plane_idx, term_idx] = term

                term_idx += 1
                if term_idx == var_per_plane:
                    plane_idx += 1
                    term_idx = 0

    # columns = ["t_appear", "t_land_early", "t_land_target", "t_land_assigned",
    #          "t_land_late", "p_land_early", "p_land_late"] + [f"sep_{i}" for i in range(n_planes)]
    # data_out = pd.DataFrame(data=data, columns=columns)
    # print(df)

    # Get upper and lower bounds for data
    lower_bounds = data.min(axis=0)
    upper_bounds = data.max(axis=0)

    return n_planes, t_freeze, data, lower_bounds, upper_bounds

def init_population(base_population, n_members=50):
    """
    Generate a randomly initialised starting population.
    """
    populations = []

    for _ in range(n_members):
        new_population = base_population.copy()
        new_population[:, COLS["T_LAND_ASSIGNED"]] = np.random.randint(
            new_population[:, COLS["T_LAND_EARLY"]],
            new_population[:, COLS["T_LAND_LATE"]],
            new_population.shape[0])
        populations.append(new_population)

    return np.array(populations)
