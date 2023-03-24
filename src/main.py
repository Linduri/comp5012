"""
Schedules planes based on set criteria
"""

import pathlib

from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

import matplotlib.pyplot as plt

from data_loader import load_data, COLS

# ====================================================== LOAD DATA

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

n_planes, t_freeze, df, lower_bounds, upper_bounds = load_data(filepath)
print(df)
# ==================================================== TRAIN MODEL

N_OBJECTIVES = 2
N_CONSTRAINTS = 0

class PlaneProblem(ElementwiseProblem):
    """
    Defines the plane problem
    """
    def __init__(self, n_vars, xl, xu):
        super().__init__(n_var=n_vars, n_obj=2, n_ieq_constr=N_CONSTRAINTS, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates how good each population member is
        """
        t_delta = x[COLS["T_LAND_TARGET"]] - x[COLS["T_LAND_ASSIGNED"]]
        early_score = t_delta*x[COLS["P_LAND_EARLY"]] if t_delta > 0 else 0
        late_score = abs(t_delta)*x[COLS["P_LAND_LATE"]] if t_delta < 0 else 0
        
        out["F"] = [early_score, late_score]
        # out["G"] =

# class ArchiveCallback(Callback):

#     def __init__(self) -> None:
#         super().__init__()
#         self.n_evals = []
#         self.opt = []

#     def notify(self, algorithm):
#         self.n_evals.append(algorithm.evaluator.n_eval)
#         self.opt.append(algorithm.opt[0].F)



problem = PlaneProblem(df.shape[1], lower_bounds, upper_bounds)

# algorithm = NSGA2(pop_size=n_planes)
# callback = ArchiveCallback()
# res = minimize(PlaneProblem(xl, xu),
#                algorithm,
#                callback=callback,
#                termination=('n_gen', 200),
#                seed=1,
#                verbose=False)

# plot = Scatter()
# # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.show()

# plt.title("Convergence")
# plt.plot(callback.n_evals, callback.opt, "--")
# plt.yscale("log")
# plt.show()