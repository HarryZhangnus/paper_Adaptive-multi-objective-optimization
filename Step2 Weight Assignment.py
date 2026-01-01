"""
Step 2: Adaptive Priority Weight Assignment - Core Framework

This calculates adaptive priority weights for multi-objective optimization based on
performance indicators using KDE-based percentiles and adaptive weighting scheme.

You need to:
1. Define your performance indicators (e.g., UTCI, RP, TQ)
2. Specify indicator directions (positive/negative)
3. Tune the steepness parameter for weight function
4. Apply weights to select optimization solutions

Repository: [https://github.com/HarryZhang386/paper_Adaptive-multi-objective-optimization.git]
"""

import numpy as np
from scipy.stats import gaussian_kde


# ==================================================================
# CORE: Weight Calculation Engine
# ==================================================================
class PriorityWeightCalculator:
    """
    Calculate adaptive priority weights using percentile-based transformation.

    Key concept:
    - Areas with worse performance get higher weights (higher priority)
    - Areas with better performance get lower weights (lower priority)
    - This enables adaptive optimization focusing on problematic areas

    Adaptive weight function: w(x) = (1-x)^k
    - x: percentile value [0, 1]
    - k: steepness parameter (controls how adaptive the weighting is)
    - Higher k → more uniform weights across all areas
    - Lower k → more focused on worst-performing areas
    """

    def __init__(self, steepness=0.56):
        """
        Parameters:
            steepness (float): Exponent k in weight function (1-x)^k
                              Proposed k value: k=0.56
                              However, please feel free to use other k values to suit your research context

        """
        self.steepness = steepness

    def weight_function(self, x):
        """
        CORE ADAPTIVE WEIGHTING FUNCTION: w(x) = (1-x)^k

        This function adaptively transforms percentiles into priority weights:
        - Input x near 0 (worst performance) → output near 1 (high priority)
        - Input x near 1 (best performance) → output near 0 (low priority)

        The steepness parameter k controls how much we focus on problem areas.

        Parameters:
            x (array): Percentile values [0, 1]

        Returns:
            array: Adaptive priority weight values [0, 1]
        """
        return np.clip(np.power(1 - x, self.steepness), 0, 1)

    def calculate_kde_percentiles(self, data, bandwidth='scott'):
        """
        Calculate smooth percentiles using Kernel Density Estimation.

        Why KDE instead of simple ranking?
        - Ranking: discrete, sensitive to outliers
        - KDE: smooth, captures underlying distribution better

        This is the CORE method for converting raw values to percentiles.

        Parameters:
            data (array): Raw indicator values
            bandwidth (str or float): KDE bandwidth parameter
                                     'scott' (default) or 'silverman'

        Returns:
            array: Smooth percentile values [0, 1]
        """
        data = np.array(data).flatten()

        # Fit KDE to data distribution
        kde = gaussian_kde(data, bw_method=bandwidth)

        percentiles = np.zeros_like(data)

        # Create dense grid for integration
        x_full = np.linspace(data.min(), data.max(), 1000)
        total_area = np.trapz(kde(x_full), x_full)

        # Calculate cumulative probability for each data point
        for i, value in enumerate(data):
            x_range = np.linspace(data.min(), value, 1000)
            prob_density = kde(x_range)
            percentile = np.trapz(prob_density, x_range) / total_area
            percentiles[i] = percentile

        return np.clip(percentiles, 0, 1)

    def calculate_weights(self, indicator_data, direction='negative'):
        """
        MAIN FUNCTION: Calculate priority weights for one indicator.

        Workflow:
        1. Convert raw values to percentiles (via KDE)
        2. Transform based on direction (positive/negative)
        3. Apply weight function to get final weights

        Parameters:
            indicator_data (array): Raw indicator values
            direction (str): 'negative' or 'positive'
                - 'negative': higher value = worse = higher weight
                  Example: thermal stress (UTCI), pollution
                - 'positive': higher value = better = lower weight
                  Example: comfort level, view ratio

        Returns:
            dict: {
                'percentiles_raw': original percentiles,
                'percentiles_transformed': after direction adjustment,
                'weights': final priority weights
            }
        """
        # Step 1: Calculate percentiles using KDE
        percentiles_raw = self.calculate_kde_percentiles(indicator_data)

        # Step 2: Transform based on direction
        if direction == 'negative':
            # Negative indicator: invert so higher values get higher weights
            # Example: UTCI 35°C (hot, bad) should get high weight
            percentiles_for_weight = 1 - percentiles_raw
        elif direction == 'positive':
            # Positive indicator: keep as-is so higher values get lower weights
            # Example: RP 0.9 (good view) should get low weight
            percentiles_for_weight = percentiles_raw
        else:
            raise ValueError("direction must be 'negative' or 'positive'")

        # Step 3: Apply sigmoid weight function
        weights = self.weight_function(percentiles_for_weight)

        return {
            'percentiles_raw': percentiles_raw,
            'percentiles_transformed': percentiles_for_weight,
            'weights': weights
        }

    def calculate_combined_weight(self, indicators_dict):
        """
        Calculate combined weight from multiple indicators.

        Parameters:
            indicators_dict (dict): {
                'indicator_name': {
                    'data': array of values,
                    'direction': 'positive' or 'negative'
                }
            }

        Returns:
            dict: {
                'indicator_name_weight': weights for each indicator,
                'combined_weight': averaged weight across all indicators
            }

        Example:
            indicators = {
                'UTCI': {'data': utci_values, 'direction': 'negative'},
                'RP': {'data':RP_values, 'direction': 'positive'},
                TQ': {'data': TQ_values, 'direction': 'positive'}
            }

            result = calculator.calculate_combined_weight(indicators)
            combined_weights = result['combined_weight']
        """
        results = {}
        individual_weights = []

        for name, config in indicators_dict.items():
            # Calculate weights for this indicator
            weight_result = self.calculate_weights(
                config['data'],
                config['direction']
            )

            # Store results
            results[f'{name}_percentile'] = weight_result['percentiles_raw']
            results[f'{name}_weight'] = weight_result['weights']

            individual_weights.append(weight_result['weights'])

        # Combine weights using arithmetic mean
        results['combined_weight'] = np.mean(individual_weights, axis=0)

        return results


# ==================================================================
# Usage Example
# ==================================================================
"""
WORKFLOW TO USE THIS FRAMEWORK:

1. PREPARE YOUR INDICATOR DATA
   - Collect performance indicators for all samples

2. DEFINE INDICATOR DIRECTIONS
   - 'negative': worse when higher (e.g., heat stress, pollution)
   - 'positive': better when higher (e.g., comfort, view quality)

3. CALCULATE WEIGHTS
   - Use PriorityWeightCalculator to compute weights
   - Tune steepness parameter if needed (0.5-0.7 recommended)

4. APPLY WEIGHTS TO OPTIMIZATION
   - Use combined_weight to filter Pareto solutions
   - Higher weight → higher priority → select those solutions
   - Can use threshold (e.g., top 20%) or ranking

Example code:

    import pandas as pd

    # Load your data (output from Step 1 optimization)
    data = pd.read_excel("optimization_results.xlsx")

    # Initialize calculator
    calculator = PriorityWeightCalculator(steepness=0.56)

    # Define your indicators
    indicators = {
        'UTCI': {
            'data': data['UTCI_orig_pred'].values,
            'direction': 'negative'  # Higher UTCI = worse = higher weight
        },
        'RP': {
            'data': data['RP_orig_pred'].values,
            'direction': 'positive'  # Higher RP = better = lower weight
        },
        'TQ': {
            'data': data['TQ_orig_pred'].values,
            'direction': 'positive'  # Higher TQ = better = lower weight
        }
    }

    # Calculate weights
    weight_results = calculator.calculate_combined_weight(indicators)

    # Add weights to dataframe
    for key, values in weight_results.items():
        data[key] = values

    # Select high-priority solutions (e.g., top 20%)
    threshold = np.percentile(weight_results['combined_weight'], 80)
    priority_samples = data[data['combined_weight'] >= threshold]

    print(f"Selected {len(priority_samples)} high-priority samples")

    # Save results
    data.to_excel("results_with_weights.xlsx", index=False)


PARAMETER TUNING GUIDE:

Direction interpretation:
- 'negative' indicators: pollution, heat stress, noise, cost
  → Higher values are BAD → Need higher priority (weight)

- 'positive' indicators: comfort, views, green space, satisfaction
  → Higher values are GOOD → Need lower priority (weight)
"""