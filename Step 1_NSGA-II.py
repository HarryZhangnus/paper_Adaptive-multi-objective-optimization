"""
Multi-Objective Optimization Framework using NSGA-II and Machine Learning Models

This framework provides a generalized approach for multi-objective optimization problems
where objectives are predicted by pre-trained machine learning models.

Key Features:
- Supports any number of decision variables and objectives
- Configurable constraints (sum range, product limits, non-decreasing variables)
- Compatible with various ML models (XGBoost, RandomForest, etc.)
- Flexible NSGA-II parameter configuration

Author: [Xudong Zhang]
Repository:
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
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

warnings.filterwarnings("ignore")


# ==================================================================
# Configuration Class
# ==================================================================
class OptimizationConfig:
    """
    Configuration class for optimization parameters.

    This class centralizes all configurable parameters, making it easy to
    modify settings without changing the core code.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from file or use defaults.

        Parameters
        ----------
        config_file : str, optional
            Path to JSON configuration file. If None, uses default values.
        """
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        else:
            self.set_defaults()

    def set_defaults(self):
        """Set default configuration values."""

        # File paths
        self.data_path = "input_data.csv"
        self.model_paths = [
            "model_obj1.json",
            "model_obj2.json",
            "model_obj3.json"
        ]
        self.output_path = "optimization_results.csv"

        # Decision variable names (must match column names in input data)
        self.decision_variables = ["var1", "var2", "var3", "var4", "var5"]

        # Objective names and optimization directions
        # 'maximize' or 'minimize' for each objective
        self.objectives = {
            "objective_1": "maximize",  # e.g., Restorative Potential
            "objective_2": "minimize",  # e.g., Thermal Comfort
            "objective_3": "maximize"  # e.g.,  Tranquility
        }

        # Variable bounds configuration
        self.bounds_config = {
            "type": "percentage",  # 'percentage' or 'absolute'
            "percentage_range": 0.30,  # ±30% around baseline
            "absolute_min": 0.0,
            "absolute_max": 1.0
        }

        # Constraint configuration
        self.constraints = {
            # Sum constraint: sum of selected variables must be in this range
            "sum_constraint": {
                "enabled": True,
                "variable_indices": [0, 1, 3, 4],  # Which variables to sum
                "min_value": 0.32,
                "max_value": 0.93
            },

            # Product constraint: (sum of variables) * (another variable) <= max
            "product_constraint": {
                "enabled": True,
                "sum_variable_indices": [0, 1, 3, 4],  # Variables to sum
                "multiply_variable_index": 2,  # Variable to multiply
                "max_value": 0.35
            },

            # Non-decreasing constraints: specified variables cannot decrease
            "non_decreasing_constraints": {
                "enabled": True,
                "variable_indices": [1]  # e.g., tree coverage should not decrease
            }
        }

        # NSGA-II algorithm parameters
        self.nsga2_params = {
            "pop_size": 100,  # Population size
            "n_gen": 100,  # Number of generations
            "crossover_prob": 0.9,  # Crossover probability
            "crossover_eta": 15,  # Crossover distribution index
            "mutation_prob": 0.2,  # Mutation probability (per variable)
            "mutation_eta": 20,  # Mutation distribution index
            "eliminate_duplicates": True
        }

        # Random seed for reproducibility
        self.random_seed = 1234

        # Optional identifier columns to preserve in results
        # These will be copied from input data to output (e.g., location IDs)
        self.identifier_columns = ["id", "longitude", "latitude"]

    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # Update attributes from JSON
        for key, value in config_dict.items():
            setattr(self, key, value)

    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

        print(f"Configuration saved to: {config_file}")


# ==================================================================
# Multi-Objective Problem Definition
# ==================================================================
class GeneralMultiObjectiveProblem(Problem):
    """
    Generalized multi-objective optimization problem.

    This class defines the optimization problem with:
    - Arbitrary number of decision variables
    - Arbitrary number of objectives
    - Flexible constraint definitions
    - Model-based objective evaluation
    """

    def __init__(
            self,
            models: List[Any],
            base_values: np.ndarray,
            config: OptimizationConfig
    ):
        """
        Initialize the optimization problem.

        Parameters
        ----------
        models : list
            List of trained ML models (one per objective)
        base_values : np.ndarray
            Baseline values for decision variables
        config : OptimizationConfig
            Configuration object containing all settings
        """
        self.models = models
        self.base_values = base_values
        self.config = config

        # Store baseline values for non-decreasing constraints
        self.non_decreasing_baselines = {}
        if config.constraints["non_decreasing_constraints"]["enabled"]:
            indices = config.constraints["non_decreasing_constraints"]["variable_indices"]
            for idx in indices:
                self.non_decreasing_baselines[idx] = base_values[idx]

        # Calculate variable bounds based on configuration
        lower_bounds, upper_bounds = self._calculate_bounds(base_values, config.bounds_config)

        # Determine optimization directions (pymoo minimizes by default)
        self.objective_directions = []
        for obj_name, direction in config.objectives.items():
            # Store 1 for maximize (will negate), -1 for minimize (keep as is)
            self.objective_directions.append(1 if direction == "maximize" else -1)

        n_var = len(config.decision_variables)
        n_obj = len(config.objectives)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _calculate_bounds(
            self,
            base_values: np.ndarray,
            bounds_config: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate variable bounds based on configuration.

        Parameters
        ----------
        base_values : np.ndarray
            Baseline values for variables
        bounds_config : dict
            Configuration for bounds calculation

        Returns
        -------
        tuple
            (lower_bounds, upper_bounds) as numpy arrays
        """
        if bounds_config["type"] == "percentage":
            # Percentage-based bounds (e.g., ±30% around baseline)
            pct = bounds_config["percentage_range"]
            lower_bounds = np.maximum(
                bounds_config["absolute_min"],
                base_values * (1 - pct)
            )
            upper_bounds = np.minimum(
                bounds_config["absolute_max"],
                base_values * (1 + pct)
            )
        else:
            # Absolute bounds
            lower_bounds = np.full_like(base_values, bounds_config["absolute_min"])
            upper_bounds = np.full_like(base_values, bounds_config["absolute_max"])

        # Ensure upper bounds don't reach exactly 1.0 (causes issues with some algorithms)
        upper_bounds = np.minimum(upper_bounds, 0.999)

        return lower_bounds, upper_bounds

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate objectives for population X.

        Parameters
        ----------
        X : np.ndarray
            Population of decision variable vectors (shape: n_samples × n_vars)
        out : dict
            Output dictionary to store objective values
        """
        # Initialize with large penalty values (for infeasible solutions)
        F = np.full((X.shape[0], self.n_obj), 1e6)

        for i, x in enumerate(X):
            # Check if solution satisfies all constraints
            if not self._check_constraints(x):
                continue  # Keep penalty value

            try:
                # Predict objectives using models
                predictions = np.array([
                    model.predict(x.reshape(1, -1))[0]
                    for model in self.models
                ])

                # Apply optimization directions
                # Maximize → negate for minimization
                # Minimize → keep as is
                F[i] = predictions * self.objective_directions

            except Exception as e:
                # If prediction fails, keep penalty value
                continue

        out["F"] = F

    def _check_constraints(self, x: np.ndarray) -> bool:
        """
        Check if decision vector x satisfies all constraints.

        Parameters
        ----------
        x : np.ndarray
            Decision variable vector

        Returns
        -------
        bool
            True if all constraints satisfied, False otherwise
        """
        constraints = self.config.constraints

        # Constraint 1: Sum constraint
        if constraints["sum_constraint"]["enabled"]:
            indices = constraints["sum_constraint"]["variable_indices"]
            total = np.sum(x[indices])
            min_val = constraints["sum_constraint"]["min_value"]
            max_val = constraints["sum_constraint"]["max_value"]

            if not (min_val <= total <= max_val):
                return False

        # Constraint 2: Product constraint
        if constraints["product_constraint"]["enabled"]:
            sum_indices = constraints["product_constraint"]["sum_variable_indices"]
            mult_index = constraints["product_constraint"]["multiply_variable_index"]
            max_val = constraints["product_constraint"]["max_value"]

            product = np.sum(x[sum_indices]) * x[mult_index]

            if product > max_val:
                return False

        # Constraint 3: Non-decreasing constraints
        if constraints["non_decreasing_constraints"]["enabled"]:
            for idx, baseline in self.non_decreasing_baselines.items():
                if x[idx] < baseline:
                    return False

        return True


# ==================================================================
# Model Loading Utilities
# ==================================================================
class ModelLoader:
    """
    Utility class for loading machine learning models.

    Supports multiple model types and formats.
    """

    @staticmethod
    def load_xgboost(model_path: str, random_state: int = 1234) -> xgb.XGBRegressor:
        """
        Load XGBoost model from file.

        Parameters
        ----------
        model_path : str
            Path to saved model file (.json or .model)
        random_state : int
            Random seed

        Returns
        -------
        xgb.XGBRegressor
            Loaded model
        """
        model = xgb.XGBRegressor(random_state=random_state)
        model.load_model(model_path)
        return model

    @staticmethod
    def load_models(
            model_paths: List[str],
            model_type: str = "xgboost",
            random_state: int = 1234
    ) -> List[Any]:
        """
        Load multiple models from files.

        Parameters
        ----------
        model_paths : list of str
            List of paths to model files
        model_type : str
            Type of model ('xgboost', 'sklearn', etc.)
        random_state : int
            Random seed

        Returns
        -------
        list
            List of loaded models
        """
        models = []

        for i, path in enumerate(model_paths):
            print(f"  Loading model {i + 1}/{len(model_paths)}: {path}")

            if model_type == "xgboost":
                model = ModelLoader.load_xgboost(path, random_state)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            models.append(model)

        return models


# ==================================================================
# Result Calculation Utilities
# ==================================================================
def calculate_change_metrics(
        original_value: float,
        optimized_value: float
) -> Tuple[float, float]:
    """
    Calculate change metrics between original and optimized values.

    Parameters
    ----------
    original_value : float
        Original (baseline) value
    optimized_value : float
        Optimized value

    Returns
    -------
    tuple
        (percentage_change, absolute_change)
    """
    abs_change = optimized_value - original_value

    if original_value == 0:
        pct_change = np.nan
    else:
        pct_change = (abs_change / original_value) * 100

    return pct_change, abs_change


# ==================================================================
# Main Optimization Engine
# ==================================================================
class MultiObjectiveOptimizer:
    """
    Main optimizer class that orchestrates the optimization process.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer with configuration.

        Parameters
        ----------
        config : OptimizationConfig
            Configuration object
        """
        self.config = config

        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # Load data and models
        print("Loading data and models...")
        self.data = pd.read_csv(config.data_path)
        self.models = ModelLoader.load_models(
            config.model_paths,
            random_state=config.random_seed
        )

        print(f"Loaded {len(self.data)} samples")
        print(f"Loaded {len(self.models)} models")

    def optimize_single_sample(
            self,
            sample_idx: int,
            x0: np.ndarray,
            verbose: bool = False
    ) -> Optional[Dict]:
        """
        Run optimization for a single sample.

        Parameters
        ----------
        sample_idx : int
            Index of the sample in the dataset
        x0 : np.ndarray
            Baseline decision variable values
        verbose : bool
            Whether to print detailed progress

        Returns
        -------
        dict or None
            Dictionary containing optimization results, or None if failed
        """
        # Define the optimization problem
        problem = GeneralMultiObjectiveProblem(self.models, x0, self.config)

        # Configure NSGA-II algorithm
        params = self.config.nsga2_params
        algorithm = NSGA2(
            pop_size=params["pop_size"],
            crossover=SBX(
                prob=params["crossover_prob"],
                eta=params["crossover_eta"]
            ),
            mutation=PM(
                prob=params["mutation_prob"],
                eta=params["mutation_eta"]
            ),
            eliminate_duplicates=params["eliminate_duplicates"]
        )

        # Set termination criterion
        termination = get_termination("n_gen", params["n_gen"])

        try:
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                termination,
                verbose=verbose,
                seed=self.config.random_seed
            )

            # Check if feasible solutions were found
            if res.X is None or len(res.X) == 0:
                return None

            # Extract Pareto-optimal solutions
            pareto_front = res.opt
            X_nd = pareto_front.get("X")  # Decision variables
            F_nd = pareto_front.get("F")  # Objective values

            return {
                "X_pareto": X_nd,
                "F_pareto": F_nd,
                "n_solutions": len(X_nd)
            }

        except Exception as e:
            if verbose:
                print(f"Optimization failed: {e}")
            return None

    def optimize_all_samples(self) -> pd.DataFrame:
        """
        Run optimization for all samples in the dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all optimization results
        """
        print(f"\nRunning optimization for {len(self.data)} samples...")
        print(f"Using NSGA-II with population={self.config.nsga2_params['pop_size']}, "
              f"generations={self.config.nsga2_params['n_gen']}")

        # Extract decision variables from data
        X = self.data[self.config.decision_variables].values

        results = []

        for i, x0 in enumerate(X):
            print(f"  Sample {i + 1}/{len(X)}...", end=" ")

            # Run optimization
            opt_result = self.optimize_single_sample(i, x0, verbose=False)

            if opt_result is None:
                print("no feasible solutions found")
                continue

            # Get original row data
            original_row = self.data.iloc[i]

            # Predict original objective values
            original_objectives = np.array([
                model.predict(x0.reshape(1, -1))[0]
                for model in self.models
            ])

            # Process each Pareto-optimal solution
            X_pareto = opt_result["X_pareto"]
            F_pareto = opt_result["F_pareto"]

            for j, (x_opt, f_opt) in enumerate(zip(X_pareto, F_pareto)):
                # Convert minimized objectives back to original scale
                optimized_objectives = f_opt * self.config.objective_directions

                # Build result row
                result_row = {
                    "sample_id": i + 1,
                    "solution_id": j + 1,
                }

                # Add identifier columns if specified
                for id_col in self.config.identifier_columns:
                    if id_col in original_row:
                        result_row[id_col] = original_row[id_col]

                # Add original decision variables
                for k, var_name in enumerate(self.config.decision_variables):
                    result_row[f"{var_name}_original"] = x0[k]

                # Add optimized decision variables
                for k, var_name in enumerate(self.config.decision_variables):
                    result_row[f"{var_name}_optimized"] = x_opt[k]

                # Add original and optimized objectives
                obj_names = list(self.config.objectives.keys())
                for k, obj_name in enumerate(obj_names):
                    result_row[f"{obj_name}_original"] = original_objectives[k]
                    result_row[f"{obj_name}_optimized"] = optimized_objectives[k]

                    # Calculate change metrics
                    pct_change, abs_change = calculate_change_metrics(
                        original_objectives[k],
                        optimized_objectives[k]
                    )
                    result_row[f"{obj_name}_change_pct"] = pct_change
                    result_row[f"{obj_name}_change_abs"] = abs_change

                results.append(result_row)

            print(f"completed, {len(X_pareto)} Pareto solutions found")

        return pd.DataFrame(results)

    def run(self) -> Optional[pd.DataFrame]:
        """
        Execute the complete optimization workflow.

        Returns
        -------
        pd.DataFrame or None
            Results dataframe if successful, None otherwise
        """
        # Run optimization
        results_df = self.optimize_all_samples()

        if len(results_df) == 0:
            print("\nNo valid optimization results obtained.")
            return None

        # Save results
        results_df.to_csv(self.config.output_path, index=False)
        print(f"\nOptimization completed successfully!")
        print(f"Total Pareto-optimal solutions found: {len(results_df)}")
        print(f"Results saved to: {self.config.output_path}")

        return results_df


# ==================================================================
# Main Execution
# ==================================================================
def main():
    """
    Main entry point for the optimization framework.

    Usage:
    ------
    1. Create a configuration file (optional) or modify OptimizationConfig defaults
    2. Prepare input data CSV with decision variables as columns
    3. Train and save ML models for each objective
    4. Run this script

    Example:
    --------
    >>> python optimization_framework.py
    >>> python optimization_framework.py --config my_config.json
    """

    # Option 1: Load configuration from file (recommended)
    # config = OptimizationConfig("config.json")

    # Option 2: Use default configuration
    config = OptimizationConfig()

    # Optional: Save default configuration as template
    # config.save_to_file("config_template.json")

    # Initialize optimizer
    optimizer = MultiObjectiveOptimizer(config)

    # Run optimization
    results = optimizer.run()

    # Optional: Perform additional analysis
    if results is not None:
        print("\n" + "=" * 70)
        print("Summary Statistics:")
        print("=" * 70)

        obj_names = list(config.objectives.keys())
        for obj_name in obj_names:
            change_col = f"{obj_name}_change_pct"
            if change_col in results.columns:
                mean_change = results[change_col].mean()
                print(f"{obj_name}: {mean_change:+.2f}% average change")


if __name__ == "__main__":
    main()