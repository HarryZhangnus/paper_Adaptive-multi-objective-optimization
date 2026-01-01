"""
Step 1: NSGA-II Multi-Objective Optimization - Core Components

This script demonstrates the core optimization framework using NSGA-II.
Users should adapt variable definitions, constraints, and models to their research.
Repository: [https://github.com/HarryZhang386/paper_Adaptive-multi-objective-optimization.git]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import warnings

warnings.filterwarnings('ignore')


# ==================================================================
# Core Optimization Problem Definition
# ==================================================================
class MultiObjectiveProblem(Problem):
    """
    Multi-objective optimization problem using surrogate models.

    Parameters:
        models: List of trained prediction models
        base_values: Baseline variable values (numpy array)
        bounds: (lower_bounds, upper_bounds) tuple
        directions: List of 1 (maximize) or -1 (minimize) for each objective
    """

    def __init__(self, models, base_values, bounds, directions):
        self.models = models
        self.base_values = base_values
        self.directions = np.array(directions)

        lower_bounds, upper_bounds = bounds

        super().__init__(
            n_var=len(base_values),
            n_obj=len(models),
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate objectives for population X."""
        F = np.zeros((X.shape[0], self.n_obj))

        for i, x in enumerate(X):
            # Apply custom constraints if needed
            if not self._is_feasible(x):
                F[i] = 1e6  # Penalty for infeasible solutions
                continue

            try:
                # Predict using surrogate models
                predictions = np.array([
                    model.predict(x.reshape(1, -1))[0]
                    for model in self.models
                ])

                # Apply optimization direction
                F[i] = predictions * self.directions

            except Exception:
                F[i] = 1e6

        out["F"] = F

    def _is_feasible(self, x):
        """
        Define custom constraints here.
        Return True if solution is feasible, False otherwise.
        """
        # Example: sum constraint
        # if not (0.3 <= np.sum(x[[0,1,3,4]]) <= 0.9):
        #     return False

        return True


# ==================================================================
# Pareto Front Filtering
# ==================================================================
def filter_pareto_front(X, F):
    """
    Filter solutions to retain only non-dominated ones.

    Parameters:
        X: Decision variables (n_solutions × n_vars)
        F: Objective values (n_solutions × n_objectives)

    Returns:
        X_pareto, F_pareto: Non-dominated solutions
    """
    n = len(F)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        for j in range(n):
            if i == j or not is_pareto[j]:
                continue

            # Check if i is dominated by j
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_pareto[i] = False
                break

    return X[is_pareto], F[is_pareto]


# ==================================================================
# NSGA-II Optimizer
# ==================================================================
def run_nsga2_optimization(
        models,
        x_baseline,
        bounds,
        directions,
        pop_size=100,
        n_generations=100,
        crossover_prob=0.9,
        mutation_prob=0.2,
        seed=42
):
    """
    Run NSGA-II optimization.

    Parameters:
        models: List of surrogate models
        x_baseline: Baseline variable values
        bounds: (lower_bounds, upper_bounds)
        directions: [1 or -1] for each objective (1=max, -1=min)
        pop_size: Population size
        n_generations: Number of generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        seed: Random seed

    Returns:
        X_opt: Optimized decision variables (Pareto front)
        F_opt: Objective values (Pareto front)
    """
    np.random.seed(seed)

    # Define problem
    problem = MultiObjectiveProblem(models, x_baseline, bounds, directions)

    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=crossover_prob, eta=15),
        mutation=PM(prob=mutation_prob, eta=20),
        eliminate_duplicates=True
    )

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_generations),
        seed=seed,
        verbose=False
    )

    if result.X is None:
        return None, None

    # Extract Pareto solutions
    X_pareto = result.opt.get("X")
    F_pareto = result.opt.get("F")

    # Additional filtering
    X_filtered, F_filtered = filter_pareto_front(X_pareto, F_pareto)

    return X_filtered, F_filtered


# ==================================================================
# Example Usage
# ==================================================================
def example_usage():
    """Demonstration of how to use the optimization framework."""

    print("=" * 60)
    print("NSGA-II Multi-Objective Optimization Example")
    print("=" * 60)

    # 1. Load your trained models
    print("\n1. Loading models...")
    models = []
    model_files = ["model_1.json", "model_2.json", "model_3.json"]

    for path in model_files:
        model = xgb.XGBRegressor()
        # model.load_model(path)  # Uncomment to load actual models
        models.append(model)

    print(f"   Loaded {len(models)} models")

    # 2. Define baseline and bounds
    print("\n2. Setting up optimization parameters...")
    x_baseline = np.array([0.5, 0.3, 0.2, 0.4, 0.1])  # Example baseline

    # Variable bounds (±30% around baseline)
    lower_bounds = np.maximum(0.0, x_baseline * 0.7)
    upper_bounds = np.minimum(0.999, x_baseline * 1.3)
    bounds = (lower_bounds, upper_bounds)

    # Optimization directions: maximize obj1, minimize obj2, maximize obj3
    directions = [1, -1, 1]

    print(f"   Variables: {len(x_baseline)}")
    print(f"   Objectives: {len(models)}")

    # 3. Run optimization
    print("\n3. Running NSGA-II optimization...")
    X_opt, F_opt = run_nsga2_optimization(
        models=models,
        x_baseline=x_baseline,
        bounds=bounds,
        directions=directions,
        pop_size=100,
        n_generations=100,
        seed=42
    )

    if X_opt is not None:
        print(f"   Found {len(X_opt)} Pareto-optimal solutions")

        # 4. Process results
        print("\n4. Sample results:")
        print(f"   Decision variables shape: {X_opt.shape}")
        print(f"   Objective values shape: {F_opt.shape}")

        # Convert objectives back to original scale
        F_original = F_opt * np.array(directions)

        print("\n   First 3 solutions (objectives):")
        for i in range(min(3, len(F_original))):
            print(f"   Solution {i + 1}: {F_original[i]}")
    else:
        print("   No feasible solutions found")

    print("\n" + "=" * 60)
    print("Optimization completed")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()

    print("\n" + "=" * 60)
    print("To use this framework:")
    print("1. Train your surrogate models (XGBoost, etc.)")
    print("2. Define your baseline values and bounds")
    print("3. Set optimization directions for each objective")
    print("4. Customize constraints in _is_feasible() method")
    print("5. Run optimization and analyze Pareto front")
    print("=" * 60)
