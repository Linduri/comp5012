import random
import numpy as np
import pandas as pd
import re

COLS = {
    "T_APPEAR": 0,
    "T_LAND_EARLY": 1,
    "T_LAND_TARGET": 2,
    "T_LAND_ASSIGNED": 3,
    "T_LAND_LATE": 4,
    "P_LAND_EARLY": 5,
    "P_LAND_LATE": 6
}

def split_terms(line):
    """
    Split a line of numbers into a list of floats
    """
    nums = re.split('\s+', line.strip())
    return [float(num) for num in nums]

def load_data(path):
    with open(path, 'r', encoding="utf-8") as file:

        # Get the plane count and freeze time from the first line.
        next_line = file.readline()

        terms = None if not next_line else split_terms(next_line)

        n_planes = int(terms[0])
        t_freeze = terms[1]
        print(f"Plane count: {n_planes}")
        print(f"Freeze time: {t_freeze}")

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

    data[:, COLS["T_LAND_ASSIGNED"]] = np.round(random.uniform(
        data[:, COLS["T_LAND_EARLY"]], data[:, COLS["T_LAND_LATE"]]))

    columns = ["t_appear", "t_land_early", "t_land_target", "t_land_assigned",
            "t_land_late", "p_land_early", "p_land_late"] + [f"sep_{i}" for i in range(n_planes)]
    df = pd.DataFrame(data=data, columns=columns)
    # print(df)

    #Get upper and lower bounds for data
    xl = data.min(axis=0)
    xu = data.max(axis=0)

    return n_planes, t_freeze, df, xl, xu
