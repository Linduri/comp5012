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
# ================================================== DRAW SCHEDULE
def draw_planes(planes, pixel_height=20, gap_height=3):
    """
    Draw plane event times for easier analysis of the data.
    """
    _lower_bounds = planes.min(axis=0)
    _upper_bounds = planes.max(axis=0)
    width = int(_upper_bounds[COLS["T_LAND_EARLY"]] +
                _upper_bounds[COLS["T_LAND_LATE"]] - _lower_bounds[COLS["T_LAND_EARLY"]])
    row_height = pixel_height+gap_height

    image = Image.new('RGB', (width, planes.shape[0]*row_height))
    ImageDraw.floodfill(image, xy=(0, 0), value=(255, 255, 255))

    for idx, plane in enumerate(planes):
        land_window = int(plane[COLS["T_LAND_LATE"]] -
                          plane[COLS["T_LAND_EARLY"]])
        for i in range(land_window):
            # Draw top and bottom lines
            image.putpixel(
                (int(plane[COLS["T_LAND_EARLY"]]) + i - 1, (idx*row_height)+gap_height), (0, 0, 0))
            image.putpixel((int(plane[COLS["T_LAND_EARLY"]]) + i - 1,
                           (idx*row_height)+row_height-gap_height-1), (0, 0, 0))

        # Draw left (early) and right (late) lines
        for j in range(pixel_height-gap_height):
            image.putpixel(
                (int(plane[COLS["T_LAND_EARLY"]])-1, (idx*row_height)+gap_height+j), (0, 0, 0))
            image.putpixel(
                (int(plane[COLS["T_LAND_LATE"]])-1, (idx*row_height)+gap_height+j), (0, 0, 0))

        # Draw appearance time
        for j in range(pixel_height-gap_height):
            image.putpixel(
                (int(plane[COLS["T_APPEAR"]])-1, (idx*row_height)+gap_height+j), (0, 0, 0))

        # Draw appearance time whisker
        land_delay = int(plane[COLS["T_LAND_EARLY"]] - plane[COLS["T_APPEAR"]])
        for i in range(land_delay):
            image.putpixel((int(plane[COLS["T_APPEAR"]]) + i - 1,
                           (idx*row_height)+int((pixel_height+gap_height)/2)), (0, 0, 0))

        # Draw assigned time
        for j in range(pixel_height-gap_height):
            image.putpixel(
                (int(plane[COLS["T_LAND_ASSIGNED"]])-1, (idx*row_height)+gap_height+j), (0, 0, 0))

        # Draw target time
        for j in range(pixel_height-gap_height-1):
            image.putpixel((int(plane[COLS["T_LAND_TARGET"]])-1, (idx*row_height) +
                           gap_height+j), (0, 0, 0) if j % 2 == 0 else (255, 255, 255))

    image.show()
    # im.save('simplePixel.png') # or any image format

# ====================================================== LOAD DATA

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

n_planes, t_freeze, data, lower_bounds, upper_bounds = load_data(filepath)

# ========================================== INITIALISE POPULATION

print("Initialise population")
schedules = init_population(data, 3)

# for pop in populations:
#     draw_planes(pop)

# ============================================= DEFINE THE PROBLEM
class PlaneProblem(ElementwiseProblem):
    """
    Defines the plane problem
    """

    def __init__(self, xl, xu):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates how good each population member is
        """
        # print(x)
        t_delta = x[:,COLS["T_LAND_ASSIGNED"]]- x[:,COLS["T_LAND_TARGET"]]
        early_score = np.sum(np.where(t_delta < 0, t_delta*x[:,COLS["P_LAND_EARLY"]], 0))
        late_score = np.sum(np.where(t_delta > 0, t_delta*x[:,COLS["P_LAND_LATE"]], 0))

        print(early_score)
        print(late_score)

        out["F"] = [early_score, late_score]
        # out["G"] =

plane_problem = PlaneProblem(lower_bounds, upper_bounds)
# for schedule in schedules:
#     res = []
#     problem._evaluate(pop, res)
    # print(problem._evaluate(pop, res))

# ============================================ DEFINE THE MUTATION
class PlaneMutation(Mutation):
    """
    Mutates each schedule
    """
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _schedule = X.copy()

        for plane in _schedule:
            if np.random.random() < self.prob:
                plane[COLS["T_LAND_ASSIGNED"]] = np.random.uniform(
                    plane[COLS["T_LAND_EARLY"]], plane[COLS["T_LAND_LATE"]])

        return _schedule
    
plane_mutation = PlaneMutation()

# for idx, schedule in enumerate(schedules):
#     print(f"Schedule {idx} before mutation")
#     draw_planes(schedule)

#     print(f"Schedule {idx} after mutation")
#     draw_planes(plane_mutation._do(plane_problem, schedule))
    

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


# plane_problem = PlaneProblem(lower_bounds, upper_bounds)
# mutation = PlaneMutation()

# plane_algorithm = NSGA2(
#     pop_size=n_planes,
#     n_offsprings=10,
#     sampling=populations,
#     mutation = mutation
# )

# termination = get_termination("n_gen", 40)

# res = minimize(plane_problem,
#                plane_algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=False,
#                callback=ArchiveCallback())

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
