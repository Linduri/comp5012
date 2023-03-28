"""
Schedules planes based on set criteria
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from PIL import Image, ImageDraw
import pandas as pd

from data_loader import load_data, init_population, COLS
from schedule import PlaneSchedule

# ====================================================== LOAD DATA
FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

schedule = PlaneSchedule(filepath)
print(pd.DataFrame(schedule.raw()))
print(pd.DataFrame(schedule.data()))
schedule.draw_planes()

# n_planes, t_freeze, data, lower_bounds, upper_bounds = load_data(filepath)

# # ========================================== INITIALISE POPULATION

# print("Initialising population...")
# schedules = init_population(data, 3)

# # for pop in populations:
# #     draw_planes(pop)

# ============================================= DEFINE THE PROBLEM
# class PlaneProblem(ElementwiseProblem):
#     """
#     Defines the plane problem
#     """

#     def __init__(self, xl, xu):
#         super().__init__(n_var=len(xl), n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

#     def _evaluate(self, x, out, *args, **kwargs):
#         """
#         Evaluates how good each population member is
#         """
#         t_delta = x[:,COLS["T_LAND_ASSIGNED"]]- x[:,COLS["T_LAND_TARGET"]]
#         early_score = np.sum(np.where(t_delta < 0, t_delta*x[:,COLS["P_LAND_EARLY"]], 0))
#         late_score = np.sum(np.where(t_delta > 0, t_delta*x[:,COLS["P_LAND_LATE"]], 0))

#         out["F"] = [early_score, late_score]
#         # out["G"] =

# print("Initialising problem...")
# plane_problem = PlaneProblem(lower_bounds, upper_bounds)
# # for schedule in schedules:
# #     res = []
# #     problem._evaluate(pop, res)
# #     print(problem._evaluate(pop, res))

# # ============================================ DEFINE THE MUTATION
# class PlaneMutation(Mutation):
#     """
#     Mutates each schedule
#     """
#     def __init__(self, prob=1.0):
#         super().__init__()
#         self.prob = prob

#     def _do(self, problem, X, **kwargs):
#         _schedule = X.copy()

#         for plane in _schedule:
#             if np.random.random() < self.prob:
#                 plane[COLS["T_LAND_ASSIGNED"]] = np.random.uniform(
#                     plane[COLS["T_LAND_EARLY"]], plane[COLS["T_LAND_LATE"]])

#         return _schedule

# print("Initialising problem...")
# plane_mutation = PlaneMutation()

# # for idx, schedule in enumerate(schedules):
# #     print(f"Schedule {idx} before mutation")
# #     draw_planes(schedule)

# #     print(f"Schedule {idx} after mutation")
# #     draw_planes(plane_mutation._do(plane_problem, schedule))

# # =================================== DEFINE THE ARCHIVE MECHANISM
# class ArchiveCallback(Callback):
#     """
#     Record the history of the network evolution.
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.n_evals = []
#         self.opt = []
#         self.data["penalties"] = []
#         self.data["population"] = []

#     def notify(self, algorithm):
#         self.n_evals.append(algorithm.evaluator.n_eval)
#         self.opt.append(algorithm.opt[0].F)
#         self.data["penalties"].append(algorithm.pop.get("F"))
#         self.data["population"].append(algorithm.pop.get("x"))

# print("Initialising archive...")
# plane_callback = ArchiveCallback()

# # =============================================== DEFINE THE MODEL
# print("Initialising algorithm...")
# plane_algorithm = NSGA2(
#     pop_size=n_planes,
#     n_offsprings=10,
#     sampling=schedules,
#     mutation = plane_mutation
# )

# # =================================== DEFINE TERMINATION CONDITION
# print("Initialising termination...")
# plane_termination = get_termination("n_gen", 40)

# # ====================================================== RUN MODEL
# res = minimize(problem=plane_problem,
#                algorithm=plane_algorithm,
#                termination=plane_termination,
#                seed=1,
#                save_history=True,
#                verbose=False,
#                callback=plane_callback)

# # =================================================== SHOW RESULTS
# # ======================================= SHOW PENATLY PROGRESSION
# combined_early_and_late = [[-x[0]+x[1] for x in X]
#                            for X in res.algorithm.callback.data["penalties"]]
# combined_early_and_late_df = pd.DataFrame(data=combined_early_and_late, columns=[
#     f"plane_{i}" for i in range(len(combined_early_and_late[0]))])

# print("Penalty evolution")
# plt.plot(combined_early_and_late_df)
# plt.xlabel("Generation")
# plt.ylabel("Penalty (Negative is early, positive is late)")
# plt.show()

# # ==================================== SHOW POPULATION PROGRESSION
# print("End population")
# draw_planes(res.algorithm.callback.data["population"][-1])
