import numpy as np
from utils import *
from model import *
from train_utils import *
import torch
from sklearn.svm import OneClassSVM
from torch_geometric.data import Data
import warnings
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print ('Using ', device)

#One of the following datasets:
#BM-CITE  LUNG-CITE  PBMC-DOGMA  PBMC-Multiome  PBMC-TEA 
dataset = 'LUNG-CITE'

print ("Loading data from %s..." %dataset)
exp, meta = load_data(dataset)

#filter low count cells
exp = filter_low_count_cells(exp, meta, dataset, threshold=100)

unique, counts = np.unique(exp["celltype"], return_counts=True)
celltype_counts = exp['celltype'].value_counts()

#Look at the distribution of celltypes
print ('Total number of cell types: ', len(celltype_counts))


class_counts = celltype_counts.to_numpy()
print(f"Gini Index: {gini_index(class_counts):.3f}")

#Split the dataset into known and unknown classes
spliting_method = 'most_common'
known_genes, unknown_genes, known_celltypes, unknown_celltypes = split_dataset(exp, celltype_counts, unique, spliting_method)

print ('Known classes: %d' % len(known_celltypes))
print ('Unkown classes: %d' % len(unknown_celltypes))

print (known_genes.shape)
print (unknown_genes.shape)

#Move some labeled cells to unlabeled
move_percentage = 0.3
sampled_indices, known_genes, unknown_genes = move_label_ratio(known_genes, unknown_genes, move_percentage)
print ('After moving labeled to unlabeled')
print ('Known cells', known_genes.shape)
print ('Unkown cells', unknown_genes.shape)

#new celltype counts
celltype_counts = known_genes['celltype'].value_counts()

known_genes, unknown_genes, known_celltypes, unknown_celltypes, known_celltypes_names, unknown_celltypes_names = map_celltypes(known_genes, unknown_genes, known_celltypes, unknown_celltypes)
# Keep track of the unknown cells for differential analysis
unknown_cell_names = np.array(unknown_genes.index)

celltypes_k = known_genes['celltype']
known_genes = known_genes.drop('celltype', axis=1)
known_genes_size = known_genes.shape[0]

celltypes_u = unknown_genes['celltype']
unknown_genes = unknown_genes.drop('celltype', axis=1)
unknown_genes_size = unknown_genes.shape[0]

input_dim = 64
known_genes, unknown_genes = dimensionality_reduction(known_genes, unknown_genes, known_genes_size, unknown_genes_size, celltypes_k, celltypes_u, n_components=input_dim)
print ('Known cells: ', known_genes.shape)
print ('Unknown cells: ', unknown_genes.shape)

#Define all model inputs 
known_genes = known_genes.to_numpy(dtype=np.float16)
unknown_genes = unknown_genes.to_numpy(dtype=np.float16)

x_labeled = torch.from_numpy(known_genes[:,:-1]).float()
y_labeled = torch.from_numpy(known_genes[:,-1])
y_labeled = y_labeled.type(torch.LongTensor)

x_unlabeled = torch.from_numpy(unknown_genes[:,:-1]).float()
y_unlabeled = torch.from_numpy(unknown_genes[:,-1])
y_unlabeled = y_unlabeled.type(torch.LongTensor)

#Move to device
x_labeled = x_labeled.to(device)
y_labeled = y_labeled.to(device)
x_unlabeled = x_unlabeled.to(device)
y_unlabeled = y_unlabeled.to(device)

#Semi-supervised learning
hidden_size = 32
masked = False
num_epochs = 500
print ('Training semisupervised block...')  
autoencoder = train_semisupervised(x_labeled, y_labeled, x_unlabeled, y_unlabeled, device, hidden_size, masked, known_celltypes, num_epochs)


print ('Running OCCs...') 
#One Class Classification
nu = 0.0001
with torch.no_grad():
    train_encoded = autoencoder(x_labeled)[0]
    test_encoded = autoencoder(x_unlabeled)[0]

all_predictions = []

for k_class in np.unique(known_celltypes):
        
        x_train = train_encoded[y_labeled == k_class]                
        ## OC-SVM MODEL
        SVM_model = OneClassSVM(kernel='rbf', gamma='auto', nu=nu, shrinking=True)
        SVM_model.fit(x_train.cpu().detach().numpy())
        predictions = SVM_model.predict(test_encoded.cpu().detach().numpy())
        all_predictions.append(predictions)

all_predictions = np.array(all_predictions).squeeze()
all_predictions = all_predictions.T

#Find new labels
rows_to_remove, gt_pseudo_labeled, pseudo_labeled_x, pseudo_labeled_y = find_new_label(all_predictions, x_unlabeled, input_dim, y_unlabeled, known_celltypes, y_labeled, x_labeled, sampled_indices, device, True)

correct = (torch.squeeze(pseudo_labeled_y) == torch.tensor(gt_pseudo_labeled)).float().sum()
known_acc = correct / pseudo_labeled_y.shape[0]
print(f'SUPERVISED ACCURACY: {known_acc.item()}')

#---------------------------------------------------
#UNSUPERVISED BLOCK
#---------------------------------------------------

#Remove the rows from the unknown data
mask = torch.ones(x_unlabeled.size(0), dtype=torch.bool)
mask[rows_to_remove] = 0
x_unlabeled = x_unlabeled[mask]
y_unlabeled = y_unlabeled[mask]
unknown_cell_names = np.delete(unknown_cell_names, rows_to_remove, axis=0)

useGraph = True
if (useGraph):
    print ('Embedding unknown cells before clustering with GNNs...')
    adj_matrix = create_adjacency_matrix(x_unlabeled)
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
    edge_index = adj_matrix.nonzero().t().contiguous()

    x_feat = x_unlabeled
    data = Data(x=x_feat, edge_index=edge_index)
    data = data.to(device)

    in_feats = x_unlabeled.shape[1]
    n_hidden = 32
    n_layers = 2
    dropout = 0.5
    num_epochs = 500
    DGI_model = train_DGI(in_feats, n_hidden, n_layers, dropout, data, device, num_epochs)
    
    # After training, get node embeddings
    DGI_model.eval()
    with torch.no_grad():
        node_embeddings = DGI_model.encoder(data, corrupt=False)

else: 
    node_embeddings = x_unlabeled

#Clustering
component_upper_bound = 15
node_embeddings = node_embeddings.detach().numpy()
print ('Running unsupervised clustering with %d components' %component_upper_bound)
DPGMM = DPGMM(component_upper_bound, node_embeddings)

#Find significant components
weights = DPGMM.weights_
threshold = 0.05

# Count the number of significant components (weights above threshold)
significant_components = np.sum(weights >= threshold)
print ('Significant components: %d' %significant_components)
print ('Number of true classes: %d' %len(unknown_celltypes))

DPGMM = mixture.BayesianGaussianMixture(n_components=significant_components, covariance_type = 'full', max_iter=50, n_init=50,
                                                tol=1e-5, init_params='k-means++', weight_concentration_prior_type='dirichlet_process', verbose=0)
DPGMM.fit(node_embeddings)

clusts = pd.DataFrame(DPGMM.predict(node_embeddings))
cluster_probabilities = DPGMM.predict_proba(node_embeddings)

unknown_acc = cluster_acc(np.squeeze(clusts.to_numpy()), y_unlabeled.numpy())
print ('UNSUPERVISED ACCURACY', unknown_acc)

#---------------------------------------------------
#FINAL REFINMENT & RESULTS
#---------------------------------------------------

tmp = clusts + len(known_celltypes)
all_labeled_x = torch.cat((pseudo_labeled_x, x_unlabeled))
all_labeled_y = torch.cat((torch.squeeze(pseudo_labeled_y), torch.tensor(np.squeeze(tmp.to_numpy()))))

# +1 because the sample itself is included
neighbors = 10
nnn = NearestNeighbors(n_neighbors=neighbors + 1)  
nnn.fit(all_labeled_x)
distances, indices = nnn.kneighbors(all_labeled_x)

# Determine the most common cluster among the neighbors
# Exclude the first neighbor because it is the sample itself
neighbor_clusters = all_labeled_y[indices[:, 1:]]  # This slices off the first column
new_clusters = np.array([mode(neighbor_clusters[i])[0][0] for i in range(neighbor_clusters.shape[0])])
print ('TOTAL (SUP and UNSP) ACCURACY', cluster_acc(all_labeled_y.numpy(), new_clusters))