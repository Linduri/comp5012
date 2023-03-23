"""
Schedules planes based on set criteria
"""

import pathlib
import re
import pandas as pd

# ====================================================== LOAD DATA

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"
print(filepath)


def split_terms(line):
    """
    Split a line of numbers into a list of floats
    """
    nums = re.split('\s+', line.strip())
    return [float(num) for num in nums]


with open(filepath, 'r', encoding="utf-8") as file:

    # Get the plane count and freeze time from the first line.
    next_line = file.readline()

    terms = None if not next_line else split_terms(next_line)

    n_planes = int(terms[0])
    t_freeze = terms[1]
    print(f"Plane count: {n_planes}")
    print(f"Freeze time: {t_freeze}")

    file.close()
