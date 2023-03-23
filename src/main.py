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

with open(filepath, 'r', encoding="utf-8") as file:
    txt_data = file.read()

# ===================================================== PARSE DATA

class ParseStage:
    METAINFO = 1
    PLANE_ATTRIBUTES = 2
    PLANE_SEPARATION = 3

parse_stage = ParseStage.METAINFO

metainfo_count = 0
plane_attributes_count = 0
plane_separation_count = 0



for line in txt_data.splitlines():
    if not line == '':
        terms = re.split('\s+', line.strip())
        terms = [float(term) for term in terms]

        for term in terms:
            # print(term)
            if parse_stage == ParseStage.METAINFO:
                if metainfo_count == 0:
                    planes.plane_count = term
                    metainfo_count += 1
                else:
                    planes.freeze_time = term
                    parse_stage = ParseStage.PLANE_ATTRIBUTES

            elif parse_stage == ParseStage.PLANE_ATTRIBUTES:
                if plane_attributes_count == 0:
                    # print("PLANE ATTRIBUTES")
                    plane = Plane()
                    plane.attributes.appearance_time = term
                elif plane_attributes_count == 1:
                    plane.attributes.earliest_landing_time = term
                elif plane_attributes_count == 2:
                    plane.attributes.target_landing_time = term
                elif plane_attributes_count == 3:
                    plane.attributes.latest_landing_time = term
                elif plane_attributes_count == 4:
                    plane.attributes.early_landing_penalty = term
                elif plane_attributes_count == 5:
                    plane.attributes.late_landing_penalty = term
                    plane_attributes_count = -1  # Minus one to account for increment
                    parse_stageparse_stage = ParseStage.PLANE_SEPARATION
                    # print("PLANE SERPARTION")

                plane_attributes_count += 1

            elif parse_stage == ParseStage.PLANE_SEPARATION:
                plane.separation_times.append(term)
                plane_separation_count += 1

                if plane_separation_count == planes.plane_count:
                    planes.add_plane(plane)
                    plane_separation_count = 0
                    parse_stage = ParseStage.PLANE_ATTRIBUTES

            else:
                print("Invalid stage!")
