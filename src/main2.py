"""
Schedules planes based on set criteria
"""

import pathlib
import re
import pandas as pd
import numpy as np
import random
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

import matplotlib.pyplot as plt

# ====================================================== LOAD DATA

FILE_IDX = 10
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"
print(filepath)

COLS = {
    "T_APPEAR": 0,
    "T_LAND_EARLY": 1,
    "T_LAND_TARGET": 2,
    "T_LAND_ASSIGNED": 3,
    "T_LAND_LATE": 4,
    "P_LAND_EARLY": 5,
    "P_LAND_LATE": 6
}

N_OBJECTIVES = 2
N_CONSTRAINTS = 0

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

class PlaneProblem(ElementwiseProblem):
    def __init__(self, xl, xu):
        super().__init__(n_var=var_per_plane, n_obj=N_OBJECTIVES, n_ieq_constr=N_CONSTRAINTS, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        t_delta = x[COLS["T_LAND_TARGET"]] - x[COLS["T_LAND_ASSIGNED"]]
        early_score = t_delta*x[COLS["P_LAND_EARLY"]] if t_delta > 0 else 0
        late_score = abs(t_delta)*x[COLS["P_LAND_LATE"]] if t_delta < 0 else 0
        
        out["F"] = [early_score, late_score]
        # out["G"] =

class ArchiveCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []

    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)

algorithm = NSGA2(pop_size=n_planes)
callback = ArchiveCallback()
res = minimize(PlaneProblem(xl, xu),
               algorithm,
               callback=callback,
               termination=('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

plt.title("Convergence")
plt.plot(callback.n_evals, callback.opt, "--")
plt.yscale("log")
plt.show()