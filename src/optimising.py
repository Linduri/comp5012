"""Optimisation testing."""
import pathlib
import optuna
import numpy as np
from pymoo.operators.crossover.pntx import TwoPointCrossover
from schedule import PlaneSchedule
from plane_optimiser import PlaneOptimiser

FILE_IDX = 1
filepath = f"{pathlib.Path(__file__).parent.parent.absolute()}/data/airland{FILE_IDX}.txt"

#print("Loading plane data...")
plane_parameters = PlaneSchedule(filepath)
def objective(trial):
    """Test."""
    prob = trial.suggest_float("prob", 0, 1)
    plane_parameters.mutate(_prob=prob)
    assigned_times = np.random.uniform(plane_parameters.t_early(), plane_parameters.t_late())
    assigned_runway = np.ones(assigned_times.shape[0])
    starting_population = np.column_stack([assigned_times, assigned_runway])

    pop_size = trial.suggest_float("POP_SIZE", 200, 500)
    generations = trial.suggest_float("GENERATIONS", 100, 400)

    solver = PlaneOptimiser(starting_population,
                            plane_parameters,
                            pop_size,
                            generations,
                            TwoPointCrossover()
    )

    res = solver.run()

    return res.F[-1][0], res.F[-1][1]

search_space = {
                "POP_SIZE": [200,250,300,350,400,450,500], 
                "GENERATIONS": [100, 125, 150, 175, 200, 250, 300, 350, 400], 
                "prob": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                }
sampler = optuna.samplers.GridSampler(search_space)

study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="plane_test",
    load_if_exists=True,
    directions=["minimize", "minimize"],
    sampler=sampler
)
study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=1)

for item in study.best_trials:
    print(f"Best trials: {item.values} : params: {item.params}")
