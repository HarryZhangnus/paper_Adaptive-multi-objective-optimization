"""
Step 3: Weighted Pareto Solution Selector (WPSS) - Core Framework

Selects the best solution from Pareto-optimal sets using adaptive weights.
This implements the final step of adaptive multi-objective optimization.

You need to:
1. Prepare optimization results from Step 1 (Pareto solutions)
2. Prepare adaptive weights from Step 2 (priority weights)
3. Define objective directions (maximize/minimize)
4. Run selection to get the best solution for each sample

Repository: [https://github.com/HarryZhangnus/paper_Adaptive-multi-objective-optimization.git]
"""

import numpy as np


# ==================================================================
# CORE: Weighted Solution Selector
# ==================================================================
class WeightedParetoSelector:
    """
    Select best solutions from Pareto sets using adaptive weighted scoring.

    Core concept:
    - Each sample has multiple Pareto-optimal solutions (trade-offs)
    - Use adaptive weights to score each solution
    - Select the solution with highest weighted score

    This bridges multi-objective optimization with adaptive prioritization,
    enabling context-aware solution selection.
    """

    def __init__(self, objective_directions):
        """
        Parameters:
            objective_directions (dict): Optimization direction for each objective
                Format: {'objective_name': 'maximize' or 'minimize'}
                Example: {'RP': 'maximize', 'UTCI': 'minimize', 'TQ': 'maximize'}
        """
        self.directions = objective_directions

    def normalize_weights(self, weights):
        """
        Normalize weights to sum to 1.

        This ensures fair comparison across different samples even if
        raw weight magnitudes differ.

        Parameters:
            weights (dict): {objective_name: weight_value}

        Returns:
            dict: Normalized weights summing to 1
        """
        total = sum(weights.values())
        if total == 0:
            # Avoid division by zero - use equal weights
            n = len(weights)
            return {obj: 1.0 / n for obj in weights.keys()}
        return {obj: w / total for obj, w in weights.items()}

    def calculate_weighted_score(self, changes, weights):
        """
        CORE SCORING FUNCTION: Calculate weighted score for a solution.

        The score reflects how well a solution addresses priority areas:
        - Higher weight × larger improvement = higher score
        - Accounts for optimization direction (max/min)

        Formula: score = Σ(weight_i × signed_change_i)
        where signed_change considers direction:
        - maximize: positive change is good (use as-is)
        - minimize: negative change is good (flip sign)

        Parameters:
            changes (dict): {objective_name: change_value}
                           Change values from optimization (can be positive/negative)
            weights (dict): {objective_name: normalized_weight}
                           Should sum to 1

        Returns:
            float: Weighted score (higher = better solution for this context)
        """
        score = 0

        for obj_name, change in changes.items():
            weight = weights[obj_name]
            direction = self.directions[obj_name]

            if direction == 'maximize':
                # Positive change is improvement
                contribution = weight * change
            elif direction == 'minimize':
                # Negative change is improvement (flip sign)
                contribution = weight * (-change)
            else:
                raise ValueError(f"Invalid direction for {obj_name}: {direction}")

            score += contribution

        return score

    def select_best_solution(self, pareto_solutions, weights):
        """
        Select the best solution from a Pareto set for one sample.

        Workflow:
        1. Normalize weights
        2. Calculate weighted score for each Pareto solution
        3. Select solution with highest score

        Parameters:
            pareto_solutions (list of dict): List of Pareto-optimal solutions
                Each solution: {'changes': {obj: value}, 'data': {...}}
            weights (dict): Raw weights for this sample
                Format: {objective_name: weight_value}

        Returns:
            dict: Best solution with added 'weighted_score' and 'weights_used'
        """
        # Normalize weights
        norm_weights = self.normalize_weights(weights)

        # Score each solution
        best_score = -np.inf
        best_solution = None

        for solution in pareto_solutions:
            score = self.calculate_weighted_score(
                solution['changes'],
                norm_weights
            )

            if score > best_score:
                best_score = score
                best_solution = solution.copy()

        # Add metadata
        best_solution['weighted_score'] = best_score
        best_solution['weights_used'] = norm_weights

        return best_solution

    def select_for_all_samples(self, all_pareto_solutions, all_weights):
        """
        Select best solutions for multiple samples.

        Parameters:
            all_pareto_solutions (list): List of Pareto sets
                Each element: list of solutions for one sample
            all_weights (list of dict): Weights for each sample
                Each element: {objective_name: weight} for one sample

        Returns:
            list: Best solution for each sample

        Example usage:
            # For 3 samples, each with multiple Pareto solutions
            pareto_sets = [
                [sol1_sample1, sol2_sample1, sol3_sample1],  # Sample 1
                [sol1_sample2, sol2_sample2],                 # Sample 2
                [sol1_sample3, sol2_sample3, sol3_sample3, sol4_sample3]  # Sample 3
            ]

            weights_list = [
                {'RP': 0.3, 'UTCI': 0.8, 'TQ': 0.2},  # Sample 1 weights
                {'RP': 0.5, 'UTCI': 0.6, 'TQ': 0.4},  # Sample 2 weights
                {'RP': 0.2, 'UTCI': 0.9, 'TQ': 0.1}   # Sample 3 weights
            ]

            selected = selector.select_for_all_samples(pareto_sets, weights_list)
        """
        selected_solutions = []

        for i, (pareto_set, weights) in enumerate(zip(all_pareto_solutions, all_weights)):
            if len(pareto_set) == 0:
                continue

            best = self.select_best_solution(pareto_set, weights)
            best['sample_id'] = i + 1
            selected_solutions.append(best)

        return selected_solutions


# ==================================================================
# Usage Example
# ==================================================================
"""
WORKFLOW TO USE THIS FRAMEWORK:

1. PREPARE INPUTS FROM PREVIOUS STEPS
   - Step 1 output: Pareto-optimal solutions for each sample
   - Step 2 output: Adaptive weights for each sample

2. ORGANIZE DATA STRUCTURE
   - Group Pareto solutions by sample
   - Extract change values for each objective
   - Match weights to samples

3. RUN SELECTION
   - Initialize selector with objective directions
   - Call select_for_all_samples() or select_best_solution()
   - Get best solution for each sample

4. APPLY SELECTED SOLUTIONS
   - These are the recommended optimization strategies
   - Implement the selected solutions in practice

Example code:

    import pandas as pd

    # Load results from Step 1 (optimization)
    opt_results = pd.read_csv("optimization_results.csv")

    # Load weights from Step 2
    weights_df = pd.read_csv("priority_weights.csv")

    # Define objective directions
    directions = {
        'RP': 'maximize',    # Want to increase resource provision
        'UTCI': 'minimize',  # Want to decrease thermal stress
        'TQ': 'maximize'     # Want to increase thermal quality
    }

    # Initialize selector
    selector = WeightedParetoSelector(objective_directions=directions)

    # Organize Pareto solutions by sample
    # Assuming opt_results has 'sample_id', 'RP_abs_change', 'UTCI_abs_change', etc.
    pareto_sets = []
    for sample_id in opt_results['sample_id'].unique():
        sample_sols = opt_results[opt_results['sample_id'] == sample_id]

        solutions = []
        for _, row in sample_sols.iterrows():
            sol = {
                'changes': {
                    'RP': row['RP_abs_change'],
                    'UTCI': row['UTCI_abs_change'],
                    'TQ': row['TQ_abs_change']
                },
                'data': row.to_dict()  # Keep all original data
            }
            solutions.append(sol)

        pareto_sets.append(solutions)

    # Extract weights for each sample
    weights_list = []
    for _, row in weights_df.iterrows():
        weights = {
            'RP': row['RP_weight'],
            'UTCI': row['UTCI_weight'],
            'TQ': row['TQ_weight']
        }
        weights_list.append(weights)

    # Select best solutions
    selected_solutions = selector.select_for_all_samples(pareto_sets, weights_list)

    print(f"Selected {len(selected_solutions)} solutions")

    # Analyze results
    for sol in selected_solutions[:3]:  # Show first 3
        print(f"\nSample {sol['sample_id']}:")
        print(f"  Weighted score: {sol['weighted_score']:.4f}")
        print(f"  Normalized weights: {sol['weights_used']}")
        print(f"  Changes: {sol['changes']}")


KEY CONCEPTS:

1. Weighted Scoring:
   - Combines multiple objectives into single score
   - Weights reflect adaptive priorities
   - Higher priority objectives have more influence

2. Direction Handling:
   - 'maximize': positive change increases score
   - 'minimize': negative change increases score
   - Ensures all improvements contribute positively to score

3. Adaptive Selection:
   - Different samples get different weights
   - Same Pareto front, different best solutions
   - Selection adapts to local context/priorities

4. Why This Works:
   - Pareto set captures all optimal trade-offs
   - Weights guide which trade-off to choose
   - Result: context-aware, adaptive optimization
"""
