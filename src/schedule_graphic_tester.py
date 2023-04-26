"""Test the schedule viewer."""

import pathlib
from plotting import Plot
from schedule import PlaneSchedule

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

print("Loading plane data...")
plane_parameters = PlaneSchedule(filepath)

print("Initialising population...")
plane_parameters.mutate(_prob=1.0)

print("Drawing starting schedule...")
output_dir = f"{pathlib.Path(__file__).parent.parent.absolute()}/report/"
Plot().image(plane_parameters.draw_planes(), "Starting schedule", save_dir=output_dir, show=False)
