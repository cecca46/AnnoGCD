## Overview
This document provides a brief description of the functions available in the `utils.py` script.

### `load_data(dataset)`
- **Description**: Loads expression and metadata for a specified dataset from a predefined directory.
- **Parameters**:
  - `dataset` (str): The name of the dataset to load.
- **Returns**: 
  - `exp` (DataFrame): The transposed expression data with genes as columns.
  - `meta` (DataFrame): The metadata associated with the expression data.

### `filter_low_count_cells(exp, meta, dataset, threshold)`
- **Description**: Filters out cells with a count below a specified threshold and standardizes certain cell type labels depending on the dataset.
- **Parameters**:
  - `exp` (DataFrame): The expression data.
  - `meta` (DataFrame): The metadata associated with the expression data.
  - `dataset` (str): The dataset being processed.
  - `threshold` (int): The minimum count required for a cell type to be retained.
- **Returns**: 
  - `exp` (DataFrame): The filtered expression data.

### `gini_index(class_counts)`
- **Description**: Calculates the Gini index, a measure of inequality among class counts.
- **Parameters**:
  - `class_counts` (list or numpy array): The counts of instances in each class.
- **Returns**: 
  - `gini` (float): The calculated Gini index, where 0 indicates perfect equality and 1 indicates maximal inequality.

### `split_dataset(exp, celltype_counts, unique, spliting_method)`
- **Description**: Splits the dataset into known and unknown classes based on a specified method.
- **Parameters**:
  - `exp` (DataFrame): The expression data.
  - `celltype_counts` (Series): The counts of each cell type.
  - `unique` (array): Unique cell types.
  - `spliting_method` (str): The method used to split the data ('most_common', 'order', 'random').
- **Returns**: 
  - `known_genes` (DataFrame): Expression data for known cell types.
  - `unknown_genes` (DataFrame): Expression data for unknown cell types.
  - `known_celltypes` (array): Known cell types.
  - `unknown_celltypes` (array): Unknown cell types.

### `move_label_ratio(known_genes, unknown_genes, move_percentage)`
- **Description**: Moves a percentage of labeled cells to the unlabeled set based on the distribution of known cell types.
- **Parameters**:
  - `known_genes` (DataFrame): Expression data for known cell types.
  - `unknown_genes` (DataFrame): Expression data for unknown cell types.
  - `move_percentage` (float): The percentage of cells to move from labeled to unlabeled.
- **Returns**: 
  - `sampled_indices` (Index): Indices of the sampled cells.
  - `known_genes` (DataFrame): Updated expression data for known cell types.
  - `unknown_genes` (DataFrame): Updated expression data for unknown cell types.

### `map_celltypes(known_genes, unknown_genes, known_celltypes, unknown_celltypes)`
- **Description**: Maps cell types to integer labels, ensuring unique mappings for known and unknown cell types.
- **Parameters**:
  - `known_genes` (DataFrame): Expression data for known cell types.
  - `unknown_genes` (DataFrame): Expression data for unknown cell types.
  - `known_celltypes` (array): Known cell types.
  - `unknown_celltypes` (array): Unknown cell types.
- **Returns**: 
  - `known_genes` (DataFrame): Expression data with integer-labeled known cell types.
  - `unknown_genes` (DataFrame): Expression data with integer-labeled unknown cell types.
  - `known_celltypes` (Series): Integer-labeled known cell types.
  - `unknown_celltypes` (Series): Integer-labeled unknown cell types.
  - `known_celltypes_names` (array): Original names of known cell types.
  - `unknown_celltypes_names` (array): Original names of unknown cell types.

### `dimensionality_reduction(known_genes, unknown_genes, known_genes_size, unknown_genes_size, celltypes_k, celltypes_u, n_components)`
- **Description**: Reduces the dimensionality of the dataset using Principal Component Analysis (PCA).
- **Parameters**:
  - `known_genes` (DataFrame): Expression data for known cell types.
  - `unknown_genes` (DataFrame): Expression data for unknown cell types.
  - `known_genes_size` (int): The number of known gene samples.
  - `unknown_genes_size` (int): The number of unknown gene samples.
  - `celltypes_k` (Series): Integer-labeled known cell types.
  - `celltypes_u` (Series): Integer-labeled unknown cell types.
  - `n_components` (int): The number of PCA components to retain.
- **Returns**: 
  - `known_genes` (DataFrame): Reduced dimensionality known gene data.
  - `unknown_genes` (DataFrame): Reduced dimensionality unknown gene data.

### `create_adjacency_matrix(data, num_neighbors=50)`
- **Description**: Creates an adjacency matrix using the k-nearest neighbors algorithm, representing the connections between samples.
- **Parameters**:
  - `data` (array): The data to create the adjacency matrix for.
  - `num_neighbors` (int): The number of neighbors to consider for each sample.
- **Returns**: 
  - `adjacency_matrix` (numpy array): The computed adjacency matrix.

### `cluster_acc(y_pred, y_true)`
- **Description**: Calculates clustering accuracy using the Hungarian algorithm to find the optimal assignment between predicted and true labels.
- **Parameters**:
  - `y_pred` (numpy array): Predicted cluster labels.
  - `y_true` (numpy array): True labels.
- **Returns**: 
  - `accuracy` (float): Clustering accuracy.

### `DPGMM(component_upper_bound, node_embeddings)`
- **Description**: Fits a Dirichlet Process Gaussian Mixture Model (DPGMM) to the given node embeddings.
- **Parameters**:
  - `component_upper_bound` (int): The upper bound on the number of components in the mixture model.
  - `node_embeddings` (numpy array): The embeddings of nodes to cluster.
- **Returns**: 
  - `DPGMM` (BayesianGaussianMixture): The trained DPGMM model.
