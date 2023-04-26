"""Test the schedule viewer."""

import os
import pathlib
from plotting import Plot
from schedule import PlaneSchedule
import logging

logging.basicConfig(level=logging.DEBUG)

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

print("Loading plane data...")
plane_parameters = PlaneSchedule(filepath)

print("Initialising population...")
plane_parameters.mutate(_prob=1.0)

print("Drawing starting schedule...")
output_dir = f"{pathlib.Path(__file__).parent.parent.absolute()}/report/"

fig = plane_parameters.draw_planes()
fig.save(os.path.join(output_dir, 'test_im.jpg'))
