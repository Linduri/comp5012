"""
Schedules planes based on set criteria
"""
import pathlib
import random
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

print("Loading plane data...")
schedule = PlaneSchedule(filepath)

# ========================================== INITIALISE POPULATION

print("Initialising population...")
schedule.mutate(prob=1.0)
print("======================== Initial population ========================")
schedule.draw_planes()
print("====================================================================")

early_times = schedule.data()[:,schedule.COLS["T_EARLY"]]
target_times = schedule.data()[:,schedule.COLS["T_TARGET"]]
late_times = schedule.data()[:,schedule.COLS["T_LATE"]]

pop_size = 100
offspring = 10
n_planes = schedule.n_planes()

pop = []
for _ in range(pop_size):
    assigned_times = np.random.uniform(early_times, late_times)
    assigned_runway = np.ones(assigned_times.shape[0])
    zipped = np.column_stack([assigned_times, assigned_runway])
    pop.append(zipped)

shaped_population = np.array(pop)
population_shape = shaped_population.shape
starting_populaiton = shaped_population.flatten()


def draw_times(times):
    print("====================================================================")
    temp = schedule.data().copy()
    temp[:, PlaneSchedule.COLS["T_ASSIGNED"]] = times
    schedule.draw_planes(data=temp)
    print("====================================================================")

# ============================================= DEFINE THE PROBLEM

class PlaneProblem(ElementwiseProblem):
    """
    Defines the plane problem
    """

    def __init__(self, starting_population):
        super().__init__(n_var=starting_population.shape[0], n_obj=2, n_ieq_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates how good each population member is
        """

        _x = np.reshape(x, population_shape)

        # Evaluate plane ealry/lateness
        t_delta = _x[:, 0] - target_times

        early_score = np.sum(
            np.where(t_delta < 0, t_delta*early_times, 0))

        late_score = np.sum(
            np.where(t_delta > 0, t_delta*late_times, 0))

        # t_delta = _x[:, PlaneSchedule.COLS["T_ASSIGNED"]] - \
        #     _x[:, PlaneSchedule.COLS["T_TARGET"]]
        # early_score = np.sum(
        #     np.where(t_delta < 0, t_delta*_x[:, PlaneSchedule.COLS["P_EARLY"]], 0))
        # late_score = np.sum(
        #     np.where(t_delta > 0, t_delta*_x[:, PlaneSchedule.COLS["P_LATE"]], 0))

        # # Check all planes are within the landing window.
        # constraint_not_early = np.where(
        #     _x[:, PlaneSchedule.COLS["T_ASSIGNED"]] >= _x[:, PlaneSchedule.COLS["T_EARLY"]], 1, 0)
        # constraint_not_late = np.where(
        #     _x[:, PlaneSchedule.COLS["T_ASSIGNED"]] <= _x[:, PlaneSchedule.COLS["T_LATE"]], 1, 0)
        # constraint_within_landing_window = constraint_not_late * constraint_not_early
        # # constraint_all_planes_within_landing_window = np.any(constraint_within_landing_window)

        out["F"] = [early_score, late_score]
#         # out["G"] =


# print("Initialising problem...")
# plane_problem = PlaneProblem(17*10)
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
#         _schedules = X.copy()
#         _schedules = _schedules.reshape(
#             (offspring, schedule.n_planes(), schedule.n_vars()))

#         for _schedule in _schedules:
#             for _plane in _schedule:
#                 if random.random() < self.prob:
#                     _plane[PlaneSchedule.COLS["T_ASSIGNED"]] = random.uniform(
#                         _plane[PlaneSchedule.COLS["T_EARLY"]], _plane[PlaneSchedule.COLS["T_LATE"]])

#         return _schedules.reshape(X.shape)


# print("Initialising mutation...")
# plane_mutation = PlaneMutation()

# # # for idx, schedule in enumerate(schedules):
# # #     print(f"Schedule {idx} before mutation")
# # #     draw_planes(schedule)

# # #     print(f"Schedule {idx} after mutation")
# # #     draw_planes(plane_mutation._do(plane_problem, schedule))

# # =================================== DEFINE THE ARCHIVE MECHANISM


# class ArchiveCallback(Callback):
#     """
#     Record the history of the network evolution.
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.n_evals = []
#         self.opt = []
#         self.data["F"] = []
#         self.data["population"] = []

#     def notify(self, algorithm):
#         self.n_evals.append(algorithm.evaluator.n_eval)
#         self.opt.append(algorithm.opt[0].F)
#         self.data["F"].append(algorithm.pop.get("F"))
#         self.data["population"].append(algorithm.pop.get("x"))


# print("Initialising archive...")
# plane_callback = ArchiveCallback()

# # =============================================== DEFINE THE MODEL
# # reshaped = schedule.data().flatten()
# # start_population = np.array([reshaped for _ in range(pop_size)])
# # print(start_population.shape)

# print("Initialising algorithm...")
# plane_algorithm = NSGA2(
#     pop_size=pop_size,
#     n_offsprings=offspring,
#     sampling=start_population,
#     mutation=plane_mutation
# )

# # =================================== DEFINE TERMINATION CONDITION
# print("Initialising termination...")
# plane_termination = get_termination("n_gen", 1000)

# # ====================================================== RUN MODEL
# print("Minimising problem...")
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
#                            for X in res.algorithm.callback.data["F"]]

# combined_early_and_late_df = pd.DataFrame(data=combined_early_and_late, columns=[
#     f"plane_{i}" for i in range(len(combined_early_and_late[0]))])

# print("Penalty evolution")
# plt.plot(combined_early_and_late_df)
# plt.xlabel("Generation")
# plt.ylabel("Penalty (Negative is early, positive is late)")
# plt.show()

# from pymoo.visualization.scatter import Scatter
# Scatter().add(res.F).show()

# # ==================================== SHOW POPULATION PROGRESSION
# print("Start population")
# schedule.draw_planes()

# print("End populations")
# best = res.X.reshape(
#             (-1, schedule.n_planes(), schedule.n_vars()))

# for _schedule in best:
#     schedule.draw_planes(data=_schedule)
