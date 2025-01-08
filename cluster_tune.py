
"""
cluster_tune.py

A script for dynamic tuning of clustering algorithms like DBSCAN and HDBSCAN
using metrics such as Silhouette Score and Davies-Bouldin Index.

Description:
    This script provides functions for:
    - Setting up parameter grids for clustering algorithms.
    - Dynamically searching for the best clustering parameters.
    - Scoring clusters based on Silhouette Score and Davies-Bouldin Index.

Example:
    from cluster_tune import cluster_tune

    best_cluster = cluster_tune(data=my_data,
                                estimator='DBSCAN',
                                metric='silhouette',
                                param_range=(0.1, 2.0),
                                steps=10,
                                verbose=True)
"""

import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import sys


# Utility functions

# setup
def setup(estimator, param_range, steps):
  """
  Set up a parameter grid for a given estimator with evenly spaced intervals.

  Parameters:
  - estimator (str): The name of the estimator (e.g., 'HDBSCAN', 'DBSCAN').
  - param_range (tuple): A tuple (min_value, max_value) defining the range.
  - steps (int): Number of intervals to divide the range into.
  Returns:
  - dict: A dictionary containing the parameter grid.
  """
  param_min, param_max = param_range

  if estimator == 'HDBSCAN':
    parameter_name = 'min_cluster_size'
    param_values = np.linspace(param_min, param_max, steps + 1).astype('int')
    param_values = np.unique(param_values)

  elif estimator == 'DBSCAN':
    parameter_name = 'eps'
    param_values = np.linspace(param_min, param_max, steps + 1).astype('float')

  param_grid = {parameter_name: param_values}

  return param_grid



def calculate_clusters(data, estimator, parameter, verbose=False):
    """
    Fit clustering algorithm and calculate basic cluster statistics.

    Parameters:
        data (DataFrame): The dataset to cluster.
        estimator (str): Clustering algorithm ("HDBSCAN" or "DBSCAN").
        parameter (float/int): Parameter for the clustering algorithm.
        verbose (bool): Whether to print detailed logs.

    Returns:
        cluster (Series): Cluster labels for each data point.
        n_clusters (int): Number of valid clusters found.
        noise (int): Number of noise points.
    """
    # Initialize the clusterer
    if estimator == 'HDBSCAN':
        clusterer = HDBSCAN(min_cluster_size=parameter)
        parameter_name = 'min_cluster_size'
    elif estimator == 'DBSCAN':
        clusterer = DBSCAN(eps=parameter)
        parameter_name = 'eps'
    else:
        raise ValueError("Unsupported estimator. Use 'DBSCAN' or 'HDBSCAN'.")

    # Fit the clusterer and assign labels
    cluster = pd.Series(clusterer.fit_predict(data)).astype(int)
    unique_clusters = [label for label in set(cluster) if label != -1]
    n_clusters = len(unique_clusters)
    noise = (cluster == -1).sum()
    noise_ratio = noise / len(data)

    # Log results
    if verbose:
        print(f"Testing {parameter_name}={parameter}")
        print(f"Noise Points={noise:.0f} ({noise_ratio:.2%}), Clusters Found={n_clusters:.0f}")

    return cluster, n_clusters, noise



def score_cluster(data, cluster, n_clusters, verbose=False):
    """
    Evaluate clustering quality using Silhouette Score and Davies-Bouldin Index.

    Parameters:
        data (DataFrame): The dataset used for clustering.
        cluster (Series): Cluster labels for each data point.
        n_clusters (int): Number of valid clusters.
        verbose (bool): Whether to print detailed logs.

    Returns:
        score (float): Silhouette Score (-1 if not computable).
        DB_index (float): Davies-Bouldin Index (Inf if not computable).
    """
    # Default values for invalid clustering
    score, DB_index = -1, np.Inf

    if n_clusters > 1:
        try:
            # Compute Silhouette Score
            score = silhouette_score(data, cluster)
        except ValueError:
            if verbose:
                print("Silhouette Score is undefined for the current clustering.")

        try:
            # Compute Davies-Bouldin Index
            DB_index = davies_bouldin_score(data, cluster)
        except ValueError:
            if verbose:
                print("Davies-Bouldin Index is undefined for the current clustering.")

    if verbose:
        print(f"Silhouette Score={score:.3f}, Davies-Bouldin Index={DB_index:.3f}")

    return score, DB_index



def find_start(data, estimator, metric, param_grid, verbose=False):
    """
    Finds the best starting parameter for smart search.

    Parameters:
        data (DataFrame): The dataset to cluster.
        estimator (object): Clustering algorithm (e.g., DBSCAN).
        metric (str): Scoring metric ("silhouette" or "davies_bouldin").
        param_grid (dict): Dictionary containing the parameter grid.
        verbose (bool): Whether to print verbose output.

    Returns:
        initial_scoring (DataFrame): DataFrame of scores and indices for each parameter.
        starting_parameter (float): The best parameter to start the search.
    """
    if verbose:
        print("Finding start point for smart search...\n")

    # Extract the parameter key and values
    param_key = list(param_grid.keys())[0]
    param_values = param_grid[param_key]

    # Track scoring results
    scoring_results = []
    cluster_results = []

    # Iterate through the parameter grid
    for parameter in param_values:
        cluster, n_cluster, noise = calculate_clusters(data, estimator, parameter, verbose)
        score, DB_index = score_cluster(data, cluster, n_cluster, verbose)
        scoring_results.append({'parameter': parameter, 'score': score, 'DB_index': DB_index})
        cluster_results.append({'parameter': parameter,  'cluster' : cluster})

    # Create DataFrame from results
    initial_scoring = pd.DataFrame(scoring_results)

    # Find the best starting parameter
    if metric == "silhouette":
        best_index = initial_scoring['score'].idxmax()
    elif metric == "davies_bouldin":
        best_index = initial_scoring['DB_index'].idxmin()
    else:
        raise ValueError("Invalid metric. Use 'silhouette' or 'davies_bouldin'.")

    # Check for edge case: no valid clusters
    if pd.isna(best_index):
        if verbose:
            print("Could not find good starting parameter. Search starts at the top end of the parameter grid.")
        starting_parameter = param_values[-1]
        best_score = -1
        best_DB_index = float('inf')
        best_cluster = None
    else:
        starting_parameter = initial_scoring.loc[best_index, 'parameter']
        best_score = initial_scoring.loc[best_index, 'score']
        best_DB_index = float(initial_scoring.loc[best_index, 'DB_index'])
        best_cluster = cluster_results[best_index]['cluster']

    if verbose:
        print(f"Initial scoring:\n{initial_scoring}")
        print(f"Starting parameter: {starting_parameter}")

    return initial_scoring, starting_parameter, best_score, best_DB_index, best_cluster



def find_direction(initial_scoring, starting_parameter, metric):
    """
    Determines the search direction based on the initial scoring grid.

    Parameters:
        initial_scoring: DataFrame with initial scoring results.
        starting_parameter: The parameter value at which the search starts.
        metric: The metric to evaluate ("silhouette" or "davies_bouldin").

    Returns:
        direction: -1 for left, 1 for right.
    """
    if metric == 'silhouette':
        # Rank scores in descending order (higher is better for silhouette)
        initial_scoring['rank'] = initial_scoring['score'].rank(ascending=False)
    elif metric == 'davies_bouldin':
        # Rank DB index in ascending order (lower is better for DB index)
        initial_scoring['rank'] = initial_scoring['DB_index'].rank(ascending=True)
    else:
        raise ValueError("Invalid metric. Use 'silhouette' or 'davies_bouldin'.")

    # Check if all scores are equal
    if metric == 'silhouette':
        all_equal = initial_scoring['score'].nunique() == 1
    elif metric == 'davies_bouldin':
        all_equal = initial_scoring['DB_index'].nunique() == 1

    if all_equal:
        # If all scores are the same, determine the direction based on starting parameter
        # Assume the starting parameter is the upper bound of the parameter range
        if starting_parameter == initial_scoring['parameter'].max():
            return -1  # Search left
        elif starting_parameter == initial_scoring['parameter'].min():
            return 1  # Search right
        else:
            raise ValueError("Starting parameter is not at a boundary for uniform scores.")

    # Otherwise, proceed as usual to find the direction based on ranked scores
    sorted_ranks = initial_scoring.sort_values(by='rank')
    first_place_index = sorted_ranks.index[0]
    second_place_index = sorted_ranks.index[1]

    # Determine the direction based on the relative position of the top two ranks
    if first_place_index > second_place_index:
        return -1  # Search left
    else:
        return 1  # Search right


def dynamic_search(data, estimator, metric, parameter, param_range, direction, steps, step_size = None,
                   verbose=False, best_parameter=None, 
                   best_score=-1, last_score=-1, best_DB_index=float('inf'), 
                   last_DB_index=float('inf'), stand_still=0, iteration=0, param_log = None, best_cluster=None):
    """
    Dynamic search for clustering optimization.

    Parameters:
        data (DataFrame): The dataset to cluster.
        estimator (str): Clustering algorithm ("DBSCAN", "HDBSCAN").
        metric (str): Scoring metric ("silhouette" or "davies_bouldin").
        parameter (float): Current parameter value being tested.
        param_range (tuple): Minimum and maximum values for the parameter.
        direction (int): Search direction (-1 for left, 1 for right).
        verbose (bool): Whether to print verbose logs.
        steps (int): Number of steps for initial grid-search.
        best_parameter (float): Best parameter found so far.
        best_score (float): Best silhouette score found so far.
        last_score (float): Last silhouette score observed.
        best_DB_index (float): Best Davies-Bouldin Index found so far.
        last_DB_index (float): Last Davies-Bouldin Index observed.
        stand_still (int): Counter for how long the search hasn't improved.
        best_cluster (Series): Cluster labels corresponding to the best parameter.

    Returns:
        best_cluster (Series): Best cluster labels found during the search.
    """
    # Unpack parameter bounds
    iteration += 1
    direction_change = False
    param_min, param_max = param_range
    parameter_type = type(parameter)
    if step_size is None:
        step_size = parameter_type(((param_max - param_min) / steps) / 2)

    # Establish minimum stepping
    if isinstance(parameter, (int, np.integer)):  # Includes numpy int types
        min_step = 1
    elif isinstance(parameter, (float, np.floating)):  # Includes numpy float types
        min_step = 0.01

    # Update parameter and check bounds
    new_parameter = parameter_type(parameter + (max(step_size, min_step) * direction))
    if new_parameter < param_min or new_parameter > param_max:
        if verbose:
            print(f"Parameter value of {new_parameter} out of bounds. Reducing step size.")
        step_size /= 2
        return dynamic_search(data, estimator, metric, parameter, param_range, 
                          direction, steps, step_size, verbose, best_parameter, 
                          best_score, last_score, best_DB_index, last_DB_index, 
                          stand_still, iteration, param_log, best_cluster)
    if new_parameter == param_min or new_parameter == param_max:
        if verbose:
            print(f"Parameter value of {new_parameter} lies on a outer bound. Changing direction.")
        direction_change = True
    
    # update parameter log
    if param_log is None: 
        param_log = []
    param_log.append(new_parameter)

    # Evaluate stopping criteria
    if (iteration > 15): 
        print("Maximum of 15 iterations reached.")
        print(f"Stop tuning at parameter = {best_parameter}, Silhouette Score: {best_score}, Davies-Bouldin Index: {best_DB_index}")
        return best_cluster 
    if (stand_still > 9):
        print("No progress made in the last 9 iterations.")
        print(f"Stop tuning at parameter = {best_parameter}, Silhouette Score: {best_score}, Davies-Bouldin Index: {best_DB_index}")
        return best_cluster
    if (len(param_log)>=5) and (new_parameter == param_log[-3] == param_log[-5]):
        print("Search is going in cirles.")
        print(f"Stop tuning at parameter = {best_parameter}, Silhouette Score: {best_score}, Davies-Bouldin Index: {best_DB_index}")
        return best_cluster        

    # Calculate clusters and scores
    cluster, n_clusters, noise = calculate_clusters(data, estimator, new_parameter, verbose)
    score, DB_index = score_cluster(data, cluster, n_clusters, verbose)

    # Metric-specific logic
    if metric == "silhouette":
        improved = score > best_score
    elif metric == "davies_bouldin":
        improved = DB_index < best_DB_index

    # Update best parameters if improved
    if improved:
        best_score = score 
        best_DB_index = DB_index 
        best_cluster = cluster
        best_parameter = new_parameter
        stand_still = 0  # Reset stand_still counter
    else:
        stand_still += 1
        direction_change = True
        step_size /= 2  # Reduce step size

    # update direction
    if direction_change:
        direction *= -1

    # Log progress
    if verbose:
        print(f"Next step: {max(step_size, min_step):.4f}, Direction: {direction}")
        print("-"*20)

    # Continue search
    return dynamic_search(data, estimator, metric, new_parameter, param_range, 
                          direction, steps, step_size, verbose, best_parameter, 
                          best_score, last_score, best_DB_index, last_DB_index, 
                          stand_still, iteration, param_log, best_cluster)


# Cluster-tune function
def cluster_tune(data, estimator, metric, param_range, steps, verbose=False):
    """
    Tune clustering parameters dynamically for the given estimator and metric.

    Parameters:
        data (DataFrame): The dataset to cluster.
        estimator (str): Clustering algorithm ("DBSCAN", "HDBSCAN").
        metric (str): Scoring metric ("silhouette" or "davies_bouldin").
        param_range (tuple): Range of parameters to tune.
        steps (int): Number of steps for initial grid-search.
        verbose (bool): Whether to print verbose logs.

    Returns:
        best_cluster (Series): Best cluster labels found during the search.
    """

    # Setup parameter grid
    param_grid = setup(estimator, param_range, steps)

    # Find starting point through grid-search
    initial_scoring, starting_parameter, best_score, best_DB_index, best_cluster = find_start(data, estimator, metric, param_grid, verbose)

    # Determine the search direction
    direction = find_direction(initial_scoring, starting_parameter, metric)
    
    # Dynamically search for optimal parameters
    best_cluster = dynamic_search(data, 
                                  estimator, 
                                  metric, 
                                  starting_parameter, 
                                  param_range, 
                                  direction, 
                                  steps,
                                  best_parameter = starting_parameter, 
                                  best_score = best_score, 
                                  best_DB_index = best_DB_index,
                                  best_cluster = best_cluster,
                                  verbose = verbose)

    # check if cluster could be found
    if best_cluster is None:
       print("Could not find any clusters. Try different settings or another algorithm.")
    
    return best_cluster



