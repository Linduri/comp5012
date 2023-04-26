"""
Schedules planes based on set criteria
"""

#marek was here 
import pathlib
import numpy as np

from PIL import Image
import pandas as pd
from pymoo.operators.crossover.pntx import TwoPointCrossover

from fpdf import FPDF
from plotting import Plot
from schedule import PlaneSchedule
from plane_optimiser import PlaneOptimiser


# ====================================================== LOAD DATA
FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

print("Loading plane data...")
plane_parameters = PlaneSchedule(filepath)

# ========================================== INITIALISE POPULATION

print("Initialising population...")
plane_parameters.mutate(_prob=1.0)

print("Parsing decision variables for evolution...")
ASSIGNED_TIMES = np.random.uniform(plane_parameters.t_early(), plane_parameters.t_late())
ASSIGNED_RUNWAY = np.ones(ASSIGNED_TIMES.shape[0])
starting_population = np.column_stack([ASSIGNED_TIMES, ASSIGNED_RUNWAY])

# =============================================== DEFINE OPTIMISER
POP_SIZE = 100
GENERATIONS = 200

solver = PlaneOptimiser(starting_population,
                        plane_parameters,
                        POP_SIZE,
                        GENERATIONS,
                        TwoPointCrossover())

# ================================================== RUN OPTIMISER
res = solver.run()

# ========================================================= REPORT
output_dir = f"{pathlib.Path(__file__).parent.parent.absolute()}/report/"

# =================================================== PLOT FIGURES
print("Drawing starting schedule...")
Plot().image(plane_parameters.draw_planes(), "Starting schedule", save_dir=output_dir, show=False)

print("Drawing best schedule...")
best = res.X[0].reshape(starting_population.shape)[:, 0]
Plot().image(plane_parameters.draw_assigned_times(best),
             "Best schedule",
             save_dir=output_dir,
             show=False)

Plot().pareto_front_2d(res.F[:, 0], res.F[:, 1],
                    save_dir=output_dir, show=False)
# Plot().normalised_pareto_front_3d(
#     res.algorithm.callback.data["F"][1:], save_dir=output_dir, show=False)
Plot().hyper_volume_2d(res.F, save_dir=output_dir, show=False)

# =================================================== GENERATE PDF
MARGIN = 10
PAGE_WIDTH = 210 - 2*MARGIN
HEADER_HEIGHT = 20
TWO_COL_WIDTH = PAGE_WIDTH/2
CELL_PADDING = 10

HYPERPARAMETERS_HEIGHT = 20

HYPERPARAMETERS_START_Y = HEADER_HEIGHT + MARGIN + CELL_PADDING
SCHEDULES_START_Y = HYPERPARAMETERS_START_Y + \
    HYPERPARAMETERS_HEIGHT + CELL_PADDING

with Image.open(f"{output_dir + 'best_schedule.png'}") as im:
    SCHEDULE_HEIGHT = (im.size[1]/im.size[0])*TWO_COL_WIDTH

with Image.open(f"{output_dir + 'pareto_front_2d.png'}") as im:
    PARETO_2D_HEIGHT = (im.size[1]/im.size[0])*TWO_COL_WIDTH

PARETO_START_Y = SCHEDULES_START_Y + CELL_PADDING + SCHEDULE_HEIGHT
HYPERVOLUME_START_Y = PARETO_START_Y + CELL_PADDING + PARETO_2D_HEIGHT

pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

pdf.set_fill_color(r=0, g=0, b=0)
pdf.set_text_color(r=255, g=255, b=255)
pdf.cell(w=0,
        h=HEADER_HEIGHT,
        txt="Plane landing optimisation",
        ln=1,
        align='C',
        fill=True)

pdf.set_text_color(r=0, g=0, b=0)
pdf.cell(w=(PAGE_WIDTH/2),
        h=HYPERPARAMETERS_HEIGHT,
        txt=f"POPULATION: {POP_SIZE}",
        align='C',
        ln=0)

pdf.cell(w=(PAGE_WIDTH/2),
        h=HYPERPARAMETERS_HEIGHT,
        txt=f"GENERATIONS: {GENERATIONS}",
        align='C',
        ln=0)

# SCHEDULES
pdf.image(output_dir + "starting_schedule.png",
        x=MARGIN, y=SCHEDULES_START_Y, w=TWO_COL_WIDTH, h=0, type='PNG')

pdf.image(output_dir + "best_schedule.png",
        x=MARGIN + TWO_COL_WIDTH, y=SCHEDULES_START_Y, w=TWO_COL_WIDTH, h=0, type='PNG')

# PARETO
pdf.image(output_dir + "pareto_front_2d.png",
        x=MARGIN, y=PARETO_START_Y, w=TWO_COL_WIDTH, h=0, type='PNG')

# pdf.image(output_dir + "pareto_front_3d.png",
#           x=MARGIN + TWO_COL_WIDTH, y=PARETO_START_Y, w=0, h=PARETO_2D_HEIGHT, type='PNG')

pdf.image(output_dir + "hypervolume.png",
        x=MARGIN, y=HYPERVOLUME_START_Y, w=TWO_COL_WIDTH, h=0, type='PNG')

# pdf.cell(w=(pw/2), h=ch, txt="Cell 2a", border=1, ln=0)
# pdf.cell(w=(pw/2), h=ch, txt="Cell 2b", border=1, ln=1)
# pdf.cell(w=(pw/3), h=ch, txt="Cell 3a", border=1, ln=0)
# pdf.cell(w=(pw/3), h=ch, txt="Cell 3b", border=1, ln=0)
# pdf.cell(w=(pw/3), h=ch, txt="Cell 3c", border=1, ln=1)
# pdf.cell(w=(pw/3), h=ch, txt="Cell 4a", border=1, ln=0)
# pdf.cell(w=(pw/3)*2, h=ch, txt="Cell 4b", border=1, ln=1)
# pdf.set_xy(x=10, y= 220) # or use pdf.ln(50)
# pdf.cell(w=0, h=ch, txt="Cell 5", border=1, ln=1)

pdf.output(output_dir + "report.pdf", 'F')

print("Done")
