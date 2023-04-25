"""
Container for an NSGA plane schedule solver
"""

import random
import logging
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.termination import get_termination


class PlaneProblem(ElementwiseProblem):
    """
    Defines the plane problem
    """
    _population_shape = None
    _plane_parameters = None

    def __init__(self, n_vars, _population_shape, _plane_parameters):
        super().__init__(n_var=n_vars, n_obj=2, n_ieq_constr=1, xl=0, xu=1)
        self._population_shape = _population_shape
        self._plane_parameters = _plane_parameters

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates how good each population member is
        """

        _x = np.reshape(x, self._population_shape)

        # Evaluate plane ealry/lateness
        t_delta = _x[:, 0] - self._plane_parameters.t_target()

        early_score = np.sum(
            np.where(t_delta < 0, -t_delta*self._plane_parameters.p_early(), 0))

        late_score = np.sum(
            np.where(t_delta > 0, t_delta*self._plane_parameters.p_late(), 0))

        out["F"] = [early_score, late_score]

        #convert np array into pandas dataframe
        _x_df=pd.DataFrame(_x,columns=["assigned_time","assigned_runway"])
        _x_df.sort_values(by=["assigned_time"],inplace=True)
        
        # for idx,plane in _x_data_frame.iloc[:-1].iterrows():
        #     plane[idx]  schedule

        G=[]
        for idx in range(_x_df.shape[0] - 1):            
            delta_time=_x_df.iloc[idx+1]["assigned_time"]-_x_df.iloc[idx]["assigned_time"]
            p1=_x_df.iloc[idx].name
            p2=_x_df.iloc[idx+1].name
        
            if delta_time > self._plane_parameters.t_separation()[p1, p2]:
               G.append(1)
            else:
               G.append(0)
            
        delta_constraint=sum(G)

        out["G"]=delta_constraint


class PlaneMutation(Mutation):
    """
    Mutates each schedule
    """
    _population_shape = None
    _plane_parameters = None

    def __init__(self, _population_shape, _plane_parameters, prob=0.5):
        super().__init__()
        self._population_shape = _population_shape
        self._plane_parameters = _plane_parameters
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _schedules = X.copy().reshape(
            (-1, self._population_shape[0], self._population_shape[1]))
        for _schedule in _schedules:
            for _idx, plane in enumerate(_schedule):
                if random.random() < self.prob:
                    plane[0] = random.uniform(
                        self._plane_parameters.t_early()[_idx],
                        self._plane_parameters.t_late()[_idx])

        return _schedules.reshape(X.shape)


class PlaneCallback(Callback):
    """
    Record the history of the network evolution.
    """
    _generation = 0
    _n_generations = 0
    _last_percent = 0

    def __init__(self, _n_generations) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
        self.data["F"] = []
        self.data["population"] = []
        self.data["F_best"] = []
        self._n_generations = _n_generations

    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)
        latest_f = algorithm.pop.get("F")
        self.data["F"].append(latest_f)
        self.data["F_best"].append(latest_f.min())
        self.data["population"].append(algorithm.pop.get("x"))

        self._generation += 1
        _new_percent = int(100*(self._generation/self._n_generations))
        if _new_percent != self._last_percent:
            print(
                f"\r{self._generation}/{self._n_generations} [{_new_percent}%]", end="")
            self._last_percent = _new_percent


class PlaneOptimiser:
    """
    An NSGA model to optimise plane landing schedules
    """
    _population_size = 500
    _generations = 500
    _starting_population = None
    _starting_population_flattened = None
    _plane_parameters = None

    _problem = None
    _mutation = None
    _callback = None
    _algorithm = None
    _crossover = None
    _termination = None

    res = None

    _logger = None

    def __init__(self, _starting_population, _plane_parameters,
                 _population_size, _generations, _crossover):
        self._starting_population = _starting_population
        self._starting_population_flattened = self._starting_population.flatten()
        self._plane_parameters = _plane_parameters
        self._population_size = _population_size
        self._generations = _generations
        self._crossover = _crossover

        self._logger = logging.getLogger(__name__)
        self._logger.info("Initialisng PlaneSolver...")

        self._logger.info("Initialisng problem...")
        self._problem = PlaneProblem(
            self._starting_population_flattened.shape[0],
            self._starting_population.shape,
            self._plane_parameters)

        self._logger.info("Initialising mutation...")
        self._mutation = PlaneMutation(
            self._starting_population.shape, self._plane_parameters)

        self._logger.info("Initialising archive...")
        self._callback = PlaneCallback(self._generations)

        self._logger.info("Initialising algorithm...")
        self._algorithm = NSGA2(
            pop_size=self._population_size,
            sampling=self._starting_population_flattened,
            mutation=self._mutation,
            crossover=self._crossover
        )

        self._logger.info("Initialising termination...")
        self._termination = get_termination("n_gen", self._generations)

    def run(self, _seed=1, _save_history=True, _verbose=False):
        """
        Run the configured optimiser.
        """

        self._logger.info("Minimising problem...")
        self.res = minimize(problem=self._problem,
                            algorithm=self._algorithm,
                            termination=self._termination,
                            seed=_seed,
                            save_history=_save_history,
                            verbose=_verbose,
                            callback=self._callback)
        self._logger.info("")

        return self.res
