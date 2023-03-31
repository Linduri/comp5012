"""
Schedules planes based on set criteria
"""
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt

from fpdf import FPDF
from PIL import Image
from pymoo.operators.crossover.pntx import TwoPointCrossover
# from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.termination import get_termination
from schedule import PlaneSchedule

# =============================================== HYPER PARAMETERS
POPULATION_SIZE = 100
GENERATIONS = 500

# ====================================================== LOAD DATA
FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

print("Loading plane data...")
schedule = PlaneSchedule(filepath)

# ========================================== INITIALISE POPULATION

print("Initialising population...")
schedule.mutate(_prob=1.0)

print("Parsing decision variables for evolution...")
ASSIGNED_TIMES = np.random.uniform(schedule.t_early(), schedule.t_late())
ASSIGNED_RUNWAY = np.ones(ASSIGNED_TIMES.shape[0])
zipped = np.column_stack([ASSIGNED_TIMES, ASSIGNED_RUNWAY])

population_shape = zipped.shape
starting_population = zipped.flatten()

# ============================================= DEFINE THE PROBLEM
class PlaneProblem(ElementwiseProblem):
    """
    Defines the plane problem
    """

    def __init__(self, n_vars):
        super().__init__(n_var=n_vars, n_obj=2, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates how good each population member is
        """

        _x = np.reshape(x, population_shape)

        # Evaluate plane ealry/lateness
        t_delta = _x[:, 0] - schedule.t_target()

        early_score = np.sum(
            np.where(t_delta < 0, -t_delta*schedule.p_early(), 0))

        late_score = np.sum(
            np.where(t_delta > 0, t_delta*schedule.p_late(), 0))

        out["F"] = [early_score, late_score]
#         # out["G"] =


print("Initialising problem...")
plane_problem = PlaneProblem(starting_population.shape[0])

# ============================================ DEFINE THE MUTATION
class PlaneMutation(Mutation):
    """
    Mutates each schedule
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _schedules = X.copy().reshape(
            (-1, population_shape[0], population_shape[1]))
        for _schedule in _schedules:
            for _idx, plane in enumerate(_schedule):
                if random.random() < self.prob:
                    plane[0] = random.uniform(
                        schedule.t_early()[_idx], schedule.t_late()[_idx])

        return _schedules.reshape(X.shape)


print("Initialising mutation...")
plane_mutation = PlaneMutation()

# =================================== DEFINE THE ARCHIVE MECHANISM
class ArchiveCallback(Callback):
    """
    Record the history of the network evolution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
        self.data["F"] = []
        self.data["population"] = []
        self.data["F_best"] = []

    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)
        latest_f = algorithm.pop.get("F")
        self.data["F"].append(latest_f)
        self.data["F_best"].append(latest_f.min())
        self.data["population"].append(algorithm.pop.get("x"))


print("Initialising archive...")
plane_callback = ArchiveCallback()

# =============================================== DEFINE THE MODEL
print("Initialising algorithm...")

plane_algorithm = NSGA2(
    pop_size=POPULATION_SIZE,
    sampling=starting_population,
    mutation=plane_mutation,
    crossover=TwoPointCrossover()
)

# =================================== DEFINE TERMINATION CONDITION
print("Initialising termination...")
plane_termination = get_termination("n_gen", GENERATIONS)

# ====================================================== RUN MODEL
print("Minimising problem...")
res = minimize(problem=plane_problem,
               algorithm=plane_algorithm,
               termination=plane_termination,
               seed=1,
               save_history=True,
               verbose=False,
               callback=plane_callback)

# # =================================================== SHOW RESULTS
# # ======================================= SHOW PENATLY PROGRESSION
# scores = np.array(res.algorithm.callback.data["F"])

# print(res.algorithm.callback.data["F"])


# combined_early_and_late = np.array([[-x[0]+x[1] for x in X]
#                            for X in res.algorithm.callback.data["F"]])



# print("Penalty evolution")
# plt.plot(combined_early_and_late_df)
# plt.xlabel("Generation")
# plt.ylabel("Penalty (Negative is early, positive is late)")
# plt.show()

# ========================================================= REPORT
output_dir = f"{pathlib.Path(__file__).parent.parent.absolute()}/report/"

# ============================================== STARTING SCHEDULE
print("Drawing starting schedule...")
fig = plt.figure()
ax = fig.subplots()
ax.imshow(schedule.draw_planes())
plt.axis('off')
plt.title("Starting schedule")
plt.savefig(output_dir + "starting_schedule.png",
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

fig.clear()

# ================================================== BEST SCHEDULE
print("Drawing best schedule...")
best = res.X[0].reshape(population_shape)[:, 0]

fig = plt.figure()
ax = fig.subplots()
ax.imshow(schedule.draw_assigned_times(best))
plt.axis('off')
plt.title("Best schedule")
plt.savefig(output_dir + "best_schedule.png",
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

fig.clear()

# ====================================================== 2D PARETO
print("Generating 2D Pareto front...")
fig = plt.figure()
ax = fig.subplots()
ax.scatter(res.F[:,0], res.F[:,1], c ="blue")
plt.title("Pareto front")
plt.xlabel(r"$F_1$")
plt.ylabel(r"$F_2$")

plt.savefig(output_dir + "pareto_front_2d.png",
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

fig.clear()

# =========================================== NORMALISED 3D PARETO
print("Generating 3D Pareto front...")
#Ignore first entry as is single starting schedule
pareto_history = np.array(res.algorithm.callback.data["F"][1:])

# Scale each pop score between zero and one
points = []
for idx, _generation in enumerate(pareto_history):
    _norm_gen = _generation

    for col in range(_generation.shape[1]):
        _norm_gen[:, col] = np.interp(_norm_gen[:, col],
        (_norm_gen[:, col].min(), _norm_gen[:, col].max()), (0, 1))

    for _member in _norm_gen:
        points.append([_member[0], _member[1], idx])

verts = np.array(points)

# Remove double
verts = np.unique(verts, axis=0)

pareto_f1 = verts[:,0]
pareto_f2 = verts[:,1]
pareto_generation = verts[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect(aspect=None, zoom=0.8)

surf = ax.plot_trisurf(pareto_f1, pareto_generation, pareto_f2, linewidth=0)

ax.invert_yaxis()

ax.set_xlabel(r"$F_1$")
ax.set_ylabel("Generation")
ax.set_zlabel(r"$F_2$")

ax.set_title("Normalised Pareto front over generations")

plt.savefig(output_dir + "pareto_front_3d.png",
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

fig.clear()

# ==================================================== HYPERVOLUME
# https://stackoverflow.com/questions/42692921/how-to-create-hypervolume-and-surface-attainment-plots-for-2-objectives-using
def plot_hyper_volume(_F, reference_point):
    print("Calculating hyper-volume...")
    # Empty pareto set
    pareto_set = np.full(_F.shape, np.inf)

    i = 0
    for point in _F:
        if i == 0:
            pareto_set[i] = point
            i += 1
        elif point[1] < pareto_set[:, 1].min():
            pareto_set[i] = point
            i += 1

    # Get rid of unused spaces
    pareto_set = pareto_set[:i + 1, :]

    # Add reference point to the pareto set
    pareto_set[i] = reference_point

    # These points will define the path to be plotted and filled
    x_path_of_points = []
    y_path_of_points = []

    for index, point in enumerate(pareto_set):

        if index < i - 1:
            plt.plot([point[0], point[0]], [point[1], pareto_set[index + 1][1]], marker='o', markersize=4, c='#4270b6',
                     mfc='black', mec='black')
            plt.plot([point[0], pareto_set[index + 1][0]], [pareto_set[index + 1][1], pareto_set[index + 1][1]],
                     marker='o', markersize=4, c='#4270b6', mfc='black', mec='black')

            x_path_of_points += [point[0], point[0], pareto_set[index + 1][0]]
            y_path_of_points += [point[1], pareto_set[index + 1][1], pareto_set[index + 1][1]]

    # Link 1 to Reference Point
    plt.plot([pareto_set[0][0], reference_point[0]], [pareto_set[0][1], reference_point[1]], marker='o', markersize=4,
             c='#4270b6', mfc='black', mec='black')
    # Link 2 to Reference Point
    plt.plot([pareto_set[-1][0], reference_point[0]], [pareto_set[-2][1], reference_point[1]], marker='o', markersize=4,
             c='#4270b6', mfc='black', mec='black')
    # Highlight the Reference Point
    plt.plot(reference_point[0], reference_point[1], 'o', color='red', markersize=8)

    # Fill the area between the Pareto set and Ref y
    plt.fill_betweenx(y_path_of_points, x_path_of_points, max(x_path_of_points) * np.ones(len(x_path_of_points)),
                      color='#dfeaff', alpha=1)

    plt.xlabel(r"$f_{\mathrm{1}}(x)$")
    plt.ylabel(r"$f_{\mathrm{2}}(x)$")
    plt.title("Hyper-volume")

    return plt

refence_point = [res.F[:,0].max(), res.F[:,1].max()]
plot_hyper_volume(res.F, refence_point).savefig(output_dir + "hypervolume.png",
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

fig.clear()

# =================================================== GENERATE PDF
MARGIN = 10
PAGE_WIDTH = 210 - 2*MARGIN
HEADER_HEIGHT = 20
TWO_COL_WIDTH = PAGE_WIDTH/2
CELL_PADDING = 10

HYPERPARAMETERS_HEIGHT = 20

HYPERPARAMETERS_START_Y = HEADER_HEIGHT + MARGIN + CELL_PADDING
SCHEDULES_START_Y = HYPERPARAMETERS_START_Y + HYPERPARAMETERS_HEIGHT + CELL_PADDING

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
    align = 'C',
    fill=True)

pdf.set_text_color(r=0, g=0, b=0)
pdf.cell(w=(PAGE_WIDTH/2),
    h=HYPERPARAMETERS_HEIGHT,
    txt=f"POPULATION: {POPULATION_SIZE}",
    align = 'C',
    ln=0)

pdf.cell(w=(PAGE_WIDTH/2),
    h=HYPERPARAMETERS_HEIGHT,
    txt=f"GENERATIONS: {GENERATIONS}",
    align = 'C',
    ln=0)

# SCHEDULES
pdf.image(output_dir + "starting_schedule.png",
          x = MARGIN, y = SCHEDULES_START_Y, w = TWO_COL_WIDTH, h = 0, type = 'PNG')

pdf.image(output_dir + "best_schedule.png",
          x = MARGIN + TWO_COL_WIDTH, y = SCHEDULES_START_Y, w = TWO_COL_WIDTH, h = 0, type = 'PNG')

# PARETO
pdf.image(output_dir + "pareto_front_2d.png",
          x = MARGIN, y = PARETO_START_Y, w = TWO_COL_WIDTH, h = 0, type = 'PNG')

pdf.image(output_dir + "pareto_front_3d.png",
          x = MARGIN + TWO_COL_WIDTH, y = PARETO_START_Y, w = 0, h = PARETO_2D_HEIGHT, type = 'PNG')

pdf.image(output_dir + "hypervolume.png",
          x = MARGIN, y = HYPERVOLUME_START_Y, w = TWO_COL_WIDTH, h = 0, type = 'PNG')

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
