from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from scipy.optimize import linear_sum_assignment


def load_data(dataset):
    data_dir = '/Users/francescoceccarelli/Library/Mobile Documents/com~apple~CloudDocs/PhD/MultiOmics/preprocessed_data'
    files = [f for f in listdir(join(data_dir, dataset)) if isfile(join(data_dir, dataset, f))]

    for file in files:
        if ('EXPR' in file):
            exp = pd.read_csv(join(data_dir, dataset, file))
            exp = exp.T
            exp.columns = exp.iloc[0]
            exp = exp.drop(exp.index[0])

        if ('META' in file): 
            meta = pd.read_csv(join(data_dir, dataset, file))
            meta.index = meta.iloc[:,0]
            meta = meta.drop(meta.columns[0], axis=1)

    assert (np.all(meta.index == exp.index))
    print ('Expression: ', exp.shape)
    print ('Meta: ', meta.shape)

    return exp, meta

def filter_low_count_cells(exp, meta, dataset, threshold):
    print ('Filtering low count cells...')
    exp['celltype'] = meta['celltype']

    if (dataset == 'BM-CITE'):
     exp["celltype"] = exp["celltype"].replace(['CD8 Effector_1', 'CD8 Effector_2'], 'CD8 Effector')
     exp["celltype"] = exp["celltype"].replace(['CD8 Memory_1', 'CD8 Memory_2'], 'CD8 Memory')
     exp["celltype"] = exp["celltype"].replace(['Prog_B 1', 'Prog_B 2'], 'Prog_B')
     exp["celltype"] = exp["celltype"].replace(['CD56 bright NK'], 'NK')
     
    if (dataset == 'LUNG-CITE'):
        exp["celltype"] = exp["celltype"].replace(['B.Plasma.1', 'B.Plasma.2'], 'B.Plasma')
        exp["celltype"] = exp["celltype"].replace(['Mono.1', 'Mono.2', 'Mono.3'], 'Mono')

    if (dataset == 'PBMC-DOGMA'):
        exp["celltype"] = exp["celltype"].replace(['CD4 Naive', 'CD4 TCM', 'CD4 TEM', ], 'CD4')
        exp["celltype"] = exp["celltype"].replace(['CD8 Naive', 'CD8 TEM', 'CD8 TCM'], 'CD8')

    value_counts = exp['celltype'].value_counts()
    values_to_remove = value_counts[value_counts < threshold].index
    exp = exp[~exp['celltype'].isin(values_to_remove)]

    return exp

def gini_index(class_counts):
    """
    Calculate the Gini index using numpy for a list of class counts.

    Args:
    class_counts (list or numpy array): A list or numpy array where each element represents the number of instances in a class.

    Returns:
    float: The Gini index, where 0 represents perfect equality (all classes have the same number of instances),
           and 1 represents maximal inequality (all instances are in one class).
    """
    # Convert class_counts to a numpy array for vectorized operations if it's not already one
    class_counts = np.array(class_counts)
    # Total number of instances
    total_instances = np.sum(class_counts)
    if total_instances == 0:
        return 0  # To handle the case where the dataset is empty and avoid division by zero

    # Calculate the probability of each class (p_i)
    class_probabilities = class_counts / total_instances
    # Calculate the Gini index using the formula: G = 1 - sum(p_i^2)
    gini = 1 - np.sum(class_probabilities ** 2)
    return gini

def split_dataset(exp, celltype_counts, unique, spliting_method):

    if (spliting_method == 'most_common'):

        known_celltypes = celltype_counts.nlargest(math.ceil(len(unique) / 2)).index
        unknown_celltypes = celltype_counts.nsmallest(math.floor(len(unique) / 2)).index

        known_genes = exp[exp['celltype'].isin(known_celltypes)]
        unknown_genes = exp[exp['celltype'].isin(unknown_celltypes)]

    if (spliting_method == 'order'):
        split_point = len(unique) // 2 if len(unique) % 2 == 0 else (len(unique) // 2) + 1

        known_celltypes = unique[:split_point]
        unknown_celltypes = unique[split_point:]

        known_genes = exp[exp['celltype'].isin(known_celltypes)]
        unknown_genes = exp[exp['celltype'].isin(unknown_celltypes)]

    if (spliting_method == 'random'):
        known_celltypes = np.random.choice(unique, math.ceil(len(unique) / 2), replace=False)
        unknown_celltypes = np.setdiff1d(unique, known_celltypes)

        known_genes = exp[exp['celltype'].isin(known_celltypes)]
        unknown_genes = exp[exp['celltype'].isin(unknown_celltypes)]

    return known_genes, unknown_genes, known_celltypes, unknown_celltypes

def move_label_ratio(known_genes, unknown_genes, move_percentage):

    #Sample the known cells and move them to the unknown set
    group_sizes = known_genes.groupby('celltype').size()
    total_size = group_sizes.sum()
    weights = group_sizes / total_size
    row_weights = known_genes['celltype'].map(weights)

    # Sample rows with the computed weights
    num_samples = int(len(known_genes) * move_percentage) 
    sampled_df = known_genes.sample(n=num_samples, weights=row_weights, replace=False)
    unknown_genes = pd.concat([unknown_genes, sampled_df])
    known_genes = known_genes.drop(sampled_df.index)
    sampled_indices = sampled_df.index

    return sampled_indices, known_genes, unknown_genes


def map_celltypes(known_genes, unknown_genes, known_celltypes, unknown_celltypes):
    # Map the cell types to integers
    # Assign integer values to known cell types in [0,..,K-1]
    # Assign integer values to unknown cell types in [K,..,N-1]

    known_types = known_genes['celltype'].unique()
    known_mapping = {cell_type: idx for idx, cell_type in enumerate(known_types)}
    known_genes['celltype'] = known_genes['celltype'].map(known_mapping)

    all_mapping = known_mapping.copy()
    next_int = len(known_mapping)

    # Assign integer values to unknown cell types
    for cell_type in unknown_genes['celltype'].unique():
        if cell_type not in all_mapping:
            all_mapping[cell_type] = next_int
            next_int += 1

    # Apply the mapping to all_df
    unknown_genes['celltype'] = unknown_genes['celltype'].map(all_mapping)

    known_celltypes_names = known_celltypes
    unknown_celltypes_names = unknown_celltypes

    known_celltypes = pd.Series(known_celltypes).map(known_mapping)
    unknown_celltypes = pd.Series(unknown_celltypes).map(all_mapping)

    inv_known_mapping = {v: k for k, v in known_mapping.items()}
    inv_all_mapping = {v: k for k, v in all_mapping.items()}

    return known_genes, unknown_genes, known_celltypes, unknown_celltypes, known_celltypes_names, unknown_celltypes_names

def dimensionality_reduction(known_genes, unknown_genes, known_genes_size, unknown_genes_size, celltypes_k, celltypes_u, n_components):

    print ('Computing PCA with %d components...' %n_components)
    pca = PCA(n_components = n_components)
    all_genes = pd.concat([known_genes, unknown_genes])
    all_genes = pd.DataFrame(pca.fit_transform(all_genes))
    known_genes = all_genes.iloc[:known_genes_size]
    unknown_genes = all_genes.iloc[known_genes_size:]

    known_genes['celltype'] = celltypes_k.tolist()
    unknown_genes['celltype'] = celltypes_u.tolist()

    return known_genes, unknown_genes

def create_adjacency_matrix(data, num_neighbors = 50):

    num_samples = data.shape[0]
    print ('Computing adjacency using %d neighbors' %num_neighbors)
    # Number of neighbors to retrieve
    num_neighbors = num_neighbors

    # Create a cKDTree from the features matrix
    kdtree = cKDTree(data)

    # List to store nearest neighbors for each point
    nearest_neighbors_list = []

    # Query for the nearest neighbors for each point in features_matrix
    for i in range(num_samples):
        query_point = data[i]
        distances, indices = kdtree.query(query_point, k=num_neighbors)
        nearest_neighbors_list.append(indices)

    adjacency_matrix = np.zeros((num_samples, num_samples))

    # Populate the adjacency matrix based on nearest neighbors list
    for i in range(num_samples):
        nearest_neighbors_indices = nearest_neighbors_list[i]
        adjacency_matrix[i, nearest_neighbors_indices] = 1

    return adjacency_matrix

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def DPGMM(component_upper_bound, node_embeddings):

    DPGMM = mixture.BayesianGaussianMixture(n_components=component_upper_bound, 
                                                max_iter=50,
                                                n_init=20,
                                                tol=1e-5, 
                                                init_params='k-means++', 
                                                weight_concentration_prior_type='dirichlet_process', verbose=0)
    DPGMM.fit(node_embeddings)
    return DPGMM