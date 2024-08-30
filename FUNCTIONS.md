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

## Overview
This document provides a brief description of the functions available in the `train_utils.py` script.

---

### `train_semisupervised(x_labeled, y_labeled, x_unlabeled, y_unlabeled, device, hidden_size, masked, known_celltypes, num_epochs)`
- **Description**: Trains a semi-supervised model using both labeled and unlabeled data. It combines an autoencoder (for unsupervised learning) and a predictor (for supervised learning).
- **Parameters**:
  - `x_labeled` (Tensor): Labeled data features.
  - `y_labeled` (Tensor): Labeled data labels.
  - `x_unlabeled` (Tensor): Unlabeled data features.
  - `y_unlabeled` (Tensor): Unlabeled data labels.
  - `device` (torch.device): The device to run the model on.
  - `hidden_size` (int): The size of the hidden layer in the model.
  - `masked` (bool): Whether to use a masked autoencoder.
  - `known_celltypes` (array): The known cell types.
  - `num_epochs` (int): The number of training epochs.
- **Returns**: 
  - `autoencoder` (Model): The trained autoencoder model.

### `find_new_label(all_predictions, x_unlabeled, input_size, y_unlabeled, known_celltypes, y_labeled, x_labeled, sampled_indices, device, print=False)`
- **Description**: Identifies new labels for unlabeled data based on predictions from multiple classifiers.
- **Parameters**:
  - `all_predictions` (array): Predictions from multiple classifiers.
  - `x_unlabeled` (Tensor): Unlabeled data features.
  - `input_size` (int): The input size.
  - `y_unlabeled` (Tensor): Unlabeled data labels.
  - `known_celltypes` (array): The known cell types.
  - `y_labeled` (Tensor): Labeled data labels.
  - `x_labeled` (Tensor): Labeled data features.
  - `sampled_indices` (array): Indices of the sampled data.
  - `device` (torch.device): The device to run the model on.
  - `print` (bool): Whether to print results.
- **Returns**: 
  - `rows_to_remove` (list): Rows to remove from the unlabeled data.
  - `gt_pseudo_labeled` (list): Ground truth pseudo-labeled data.
  - `pseudo_labeled_x` (Tensor): Features of pseudo-labeled data.
  - `pseudo_labeled_y` (Tensor): Labels of pseudo-labeled data.

### `initialize_pseudo_labeled(input_size, device)`
- **Description**: Initializes tensors for pseudo-labeled data.
- **Parameters**:
  - `input_size` (int): The input size.
  - `device` (torch.device): The device to run the model on.
- **Returns**: 
  - `pseudo_labeled_x` (Tensor): Initialized features tensor.
  - `pseudo_labeled_y` (Tensor): Initialized labels tensor.

### `handle_single_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, should_have_labeled, device)`
- **Description**: Handles the labeling of a single-class sample from the unlabeled data.
- **Parameters**:
  - `index` (int): Index of the sample.
  - `x_unlabeled` (Tensor): Unlabeled data features.
  - `col_indices` (array): Indices of the predicted class.
  - `pseudo_labeled_x` (Tensor): Features of pseudo-labeled data.
  - `pseudo_labeled_y` (Tensor): Labels of pseudo-labeled data.
  - `rows_to_remove` (list): List of rows to remove.
  - `gt_pseudo_labeled` (list): Ground truth pseudo-labeled data.
  - `y_unlabeled` (Tensor): Unlabeled data labels.
  - `known_celltypes` (array): The known cell types.
  - `should_have_labeled` (int): Counter for correct labeling.
  - `device` (torch.device): The device to run the model on.
- **Returns**: 
  - Updated versions of `pseudo_labeled_x`, `pseudo_labeled_y`, `rows_to_remove`, `gt_pseudo_labeled`, and `should_have_labeled`.

### `handle_multi_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, y_labeled, x_labeled, should_have_labeled, device)`
- **Description**: Handles the labeling of a multi-class sample from the unlabeled data.
- **Parameters**: Similar to `handle_single_class_sample`, with additional parameters:
  - `y_labeled` (Tensor): Labeled data labels.
  - `x_labeled` (Tensor): Labeled data features.
- **Returns**: 
  - Updated versions of `pseudo_labeled_x`, `pseudo_labeled_y`, `rows_to_remove`, `gt_pseudo_labeled`, and `should_have_labeled`.

### `print_results(rows_to_remove, sampled_indices, should_have_labeled)`
- **Description**: Prints the results of the labeling process.
- **Parameters**:
  - `rows_to_remove` (list): Rows removed from the unlabeled data.
  - `sampled_indices` (array): Indices of the sampled data.
  - `should_have_labeled` (int): Counter for correct labeling.

### `train_DGI(in_feats, n_hidden, n_layers, dropout, data, device, num_epochs)`
- **Description**: Trains a Deep Graph Infomax (DGI) model on the given data.
- **Parameters**:
  - `in_feats` (int): The number of input features.
  - `n_hidden` (int): The size of the hidden layer.
  - `n_layers` (int): The number of layers in the model.
  - `dropout` (float): The dropout rate.
  - `data` (Data): The data object containing features and edge indices.
  - `device` (torch.device): The device to run the model on.
  - `num_epochs` (int): The number of training epochs.
- **Returns**: 
  - `model` (DGI): The trained DGI model.

## Overview
This document provides a brief description of the functions available in the `model.py` script.

---

### `MaskedAutoEncoder`
- **Description**: A masked autoencoder model that includes an encoder, a mask predictor, and a decoder. The model can dynamically create layers based on the provided hidden sizes and includes functionality to mask input data during training.
- **Methods**:
  - `__init__(self, input_size, hidden_size=[32, 32], dropout=0.2, mask_prob=0.2)`: Initializes the MaskedAutoEncoder with the specified input size, hidden layers, dropout rate, and mask probability.
  - `contrsuct_model(self, input_size, hidden_size, dropout)`: Dynamically constructs the encoder, mask predictor, and decoder layers.
  - `forward(self, X)`: The forward pass of the model, which applies masking during training and returns the encoded representation and the reconstructed input.
  - `mask_input(self, X)`: Masks the input by randomly swapping elements within the data, based on a Bernoulli distribution.

### `Autoencoder`
- **Description**: A simple autoencoder model with an encoder and a decoder. The model uses ReLU activations for the hidden layers.
- **Methods**:
  - `__init__(self, input_size, hidden_size)`: Initializes the Autoencoder with the specified input and hidden sizes.
  - `forward(self, x)`: The forward pass of the model, returning the encoded representation and the reconstructed input.

### `Predictor`
- **Description**: A neural network model for prediction, using fully connected layers with ReLU activations. The final layer applies a softmax activation for classification.
- **Methods**:
  - `__init__(self, input_size, hidden_size, num_classes)`: Initializes the Predictor with the specified input size, hidden size, and number of output classes.
  - `forward(self, x)`: The forward pass of the model, returning the class probabilities.

### `MultiObjectiveLoss`
- **Description**: A custom loss function that computes a weighted sum of two loss components. The weights are learned during training.
- **Methods**:
  - `__init__(self)`: Initializes the MultiObjectiveLoss with random initial weights.
  - `forward(self, loss1, loss2)`: Computes the weighted sum of the two provided losses.

### `GCN`
- **Description**: A Graph Convolutional Network (GCN) model that applies graph convolutional layers to input features and edges.
- **Methods**:
  - `__init__(self, in_feats, n_hidden, n_layers, dropout)`: Initializes the GCN with the specified input features, hidden layer size, number of layers, and dropout rate.
  - `forward(self, x, edge_index)`: The forward pass of the model, applying the GCN layers to the input data and edge index.

### `Encoder`
- **Description**: A graph encoder that uses a GCN to encode node features, with optional corruption of the input data.
- **Methods**:
  - `__init__(self, in_feats, n_hidden, n_layers, dropout)`: Initializes the Encoder with the specified input features, hidden layer size, number of layers, and dropout rate.
  - `forward(self, data, corrupt=False)`: The forward pass of the encoder, with an option to corrupt the input data by shuffling the nodes.

### `Discriminator`
- **Description**: A discriminator model for contrasting positive and negative samples in the context of Deep Graph Infomax (DGI).
- **Methods**:
  - `__init__(self, n_hidden)`: Initializes the Discriminator with the specified hidden layer size.
  - `uniform(self, size, tensor)`: Initializes the weight tensor with a uniform distribution.
  - `reset_parameters(self)`: Resets the parameters of the weight matrix.
  - `forward(self, features, summary)`: The forward pass of the discriminator, which scores the agreement between node features and a summary representation.

### `DGI`
- **Description**: A Deep Graph Infomax (DGI) model that learns node embeddings by maximizing mutual information between node representations and a summary of the graph.
- **Methods**:
  - `__init__(self, in_feats, n_hidden, n_layers, dropout)`: Initializes the DGI model with the specified input features, hidden layer size, number of layers, and dropout rate.
  - `forward(self, data)`: The forward pass of the DGI model, computing the loss by contrasting positive and negative samples.
