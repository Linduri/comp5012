"""
Schedules planes based on set criteria
"""
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fpdf import FPDF
from pymoo.visualization.scatter import Scatter
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
OFFSPRING = 10
GENERATIONS = 100

# ====================================================== LOAD DATA
FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

print("Loading plane data...")
schedule = PlaneSchedule(filepath)

# ========================================== INITIALISE POPULATION

print("Initialising population...")
schedule.mutate(_prob=1.0)
print("======================== Initial population ========================")
schedule.draw_planes()
print("====================================================================")

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
            for idx, plane in enumerate(_schedule):
                if random.random() < self.prob:
                    plane[0] = random.uniform(
                        schedule.t_early()[idx], schedule.t_late()[idx])

        return _schedules.reshape(X.shape)


print("Initialising mutation...")
plane_mutation = PlaneMutation()

# # =================================== DEFINE THE ARCHIVE MECHANISM
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
    # n_offsprings=OFFSPRING,
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

# ========================================================= PARETO
print("Generating Pareto front graph")
plt.scatter(res.F[:,0], res.F[:,1], c ="blue")
plt.title("Pareto front")
plt.xlabel(r"$F_1$")
plt.ylabel(r"$F_2$")

plt.savefig('./pareto_front.png',
           transparent=False,
           facecolor='white',
           bbox_inches="tight")

# plt.show()

# ==================================== SHOW POPULATION PROGRESSION
# print("Start population")
# schedule.draw_planes()

# print("End population")
# best = res.X[0].reshape(population_shape)[:, 0]
# schedule.draw_assigned_times(best)

# =================================================== GENERATE PDF

margin = 10
# Page width: Width of A4 is 210mm
page_width = 210 - 2*margin
# Cell height
cell_height = 50
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

pdf.set_fill_color(r=0, g=0, b=0)
pdf.set_text_color(r=255, g=255, b=255)
pdf.cell(w=0,
    h=20,
    txt="Plane landing optimisation",
    ln=1,
    align = 'C',
    fill=True)

pdf.image('./pareto_front.png', 
          x = 10, y = None, w = 100, h = 0, type = 'PNG')

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

# pdf.output(f"{pathlib.Path(__file__).parent.parent.absolute()}/src/report.pdf", 'F')

print("Done")
