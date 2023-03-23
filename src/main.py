"""
Schedules planes based on set criteria
"""

import pathlib
import pandas as pd

#====================================================== LOAD DATA

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"
print(filepath)

with open(filepath, 'r', encoding="utf-8") as file:
    data = file.read()

#===================================================== PARSE DATA

