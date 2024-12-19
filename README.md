# Cluster Tuning with Dynamic Search Tuning

This repository contains a Python module for tuning clustering algorithms (DBSCAN and HDBSCAN) using dynamic search. 

## Features
The `cluster_tune` function performs the following tasks:

- **Parameter Tuning**: Dynamically refines clustering parameters (`eps` for DBSCAN and `min_cluster_size` for HDBSCAN) to maximize clustering quality.
- **Clustering Evaluation**: Evaluates clusters using metrics like **Silhouette Score** and **Davies-Bouldin Index**.
- **Smart Search**: Implements recursive search to efficiently identify optimal parameters with adaptive step size and search direction.

## Usage
This module can be imported into your projects to fine-tune clustering algorithms.

### Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/cluster-tuning.git
```

2. **Import the module:**

```python
from cluster_tune import cluster_tune
```
