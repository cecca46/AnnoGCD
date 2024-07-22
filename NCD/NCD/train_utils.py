import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score


def train_semisupervised(x_labeled, y_labeled, x_unlabeled, y_unlabeled, device, hidden_size, masked, known_celltypes, num_epochs):

    input_size = x_labeled.shape[1]
    hidden_size = hidden_size
    masked = False

    #all data for unsupervised autoencoder loss
    input_data = torch.cat((x_labeled, x_unlabeled), dim=0).to(device)

    # Instantiate the autoencoder, predictor & pseudo_predictor
    if masked:
        autoencoder = MaskedAutoEncoder(input_size).float().to(device)
        print ("Using Masked Autoencoder...")
    else:
        autoencoder = Autoencoder(input_size, hidden_size).float().to(device)

    predictor = Predictor(input_size, hidden_size, len(known_celltypes)).float().to(device)

    #losses 
    criterion_mse = nn.MSELoss() #autoencoder
    criterion_CE = nn.CrossEntropyLoss() #predictor

    # Initialize learnable weights for the three loss components
    # Start with equal weights
    loss_weights = nn.Parameter(torch.ones(2, device=device) / 2)  

    #optimizers 
    autoencoder_optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001)
    predictor_optimizer = optim.AdamW(predictor.parameters(), lr=0.001)
    optimizer_weights = optim.AdamW([loss_weights], lr=0.001)

    num_epochs = num_epochs

    for epoch in range(num_epochs):
    
        # Autoencoder (unsupervised loss)
        encoded, reconstructed = autoencoder(input_data)
        autoencoder_loss = criterion_mse(reconstructed, input_data)
        
        #Predictor labeled (supervised loss)
        predictions = predictor(x_labeled)
        predictor_loss = criterion_CE(predictions, y_labeled)
            
        # Compute the weighted sum of the loss terms
        loss_weights_normalized = F.softmax(loss_weights, dim=0)
        total_loss = (loss_weights_normalized[0] * autoencoder_loss +
                            loss_weights_normalized[1] * predictor_loss)
                    
        # Backpropagation and optimization
        autoencoder_optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        optimizer_weights.zero_grad()
            
        total_loss.backward()

        autoencoder_optimizer.step()
        predictor_optimizer.step()
        optimizer_weights.step()
                
        if ((epoch % 50) == 0):
            print(f'Epoch [{epoch}/{num_epochs}], Total Loss: {total_loss.item():.3f}, Autoencoder Loss: {autoencoder_loss.item():.3f}, Predictor Loss: {predictor_loss:.3f}')
            print (f'Autoencoder Weights: {loss_weights_normalized[0].item():.3f}, Predictor Weights: {loss_weights_normalized[1].item():.3f}')

    return autoencoder

def find_new_label(all_predictions, x_unlabeled, input_size, y_unlabeled, known_celltypes, y_labeled, x_labeled, sampled_indices, device, print=False):

    rows_to_remove = []
    gt_pseudo_labeled = []
    should_have_labeled = 0

    pseudo_labeled_x, pseudo_labeled_y = initialize_pseudo_labeled(input_size, device)
    
    for index in range(x_unlabeled.shape[0]):
        col_indices = np.where(all_predictions[index] == 1)[0]

        if len(col_indices) == 0:
            continue
                
        if len(col_indices) == 1:
            pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, should_have_labeled = \
                handle_single_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, should_have_labeled, device)
        elif len(col_indices) > 1:
            pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, should_have_labeled = \
                handle_multi_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, y_labeled, x_labeled, should_have_labeled, device)
    
    if (print):
        print_results(rows_to_remove, sampled_indices, should_have_labeled)

    return rows_to_remove, gt_pseudo_labeled, pseudo_labeled_x, pseudo_labeled_y

def initialize_pseudo_labeled(input_size, device):
    pseudo_labeled_x = torch.empty((0, input_size), dtype=torch.float32).to(device)
    pseudo_labeled_y = torch.empty((0, 1), dtype=torch.long).to(device)
    return pseudo_labeled_x, pseudo_labeled_y

def handle_single_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, should_have_labeled, device):
    new_labeled_sample = np.expand_dims(x_unlabeled[index].cpu().detach().numpy(), axis=0)
    new_label = col_indices.reshape(1, 1)

    rows_to_remove.append(index)
    pseudo_labeled_x = torch.cat((pseudo_labeled_x, torch.from_numpy(new_labeled_sample).to(device)), 0)
    pseudo_labeled_y = torch.cat((pseudo_labeled_y, torch.from_numpy(new_label).to(device)), 0)
    gt_pseudo_labeled.append(y_unlabeled[index])
    
    if y_unlabeled[index].cpu().detach().numpy() in np.array(known_celltypes):
        should_have_labeled += 1
    
    return pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, should_have_labeled

def handle_multi_class_sample(index, x_unlabeled, col_indices, pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, y_unlabeled, known_celltypes, y_labeled, x_labeled, should_have_labeled, device):
    selector = np.isin(y_labeled.cpu().detach().numpy(), col_indices)
    tmp_x = x_labeled[selector].cpu().detach().numpy()
    tmp_y = y_labeled[selector].cpu().detach().numpy()

    n_neighbors = 20
    nearNei = NearestNeighbors(n_neighbors=n_neighbors).fit(tmp_x)
    distances, indices = nearNei.kneighbors(x_unlabeled[index].cpu().detach().numpy().reshape(1, -1))
    nearest_labels = tmp_y[indices[0]]

    label_counter = Counter(nearest_labels)
    most_common_label, count = label_counter.most_common(1)[0]

    new_labeled_sample = np.expand_dims(x_unlabeled[index].cpu().detach().numpy(), axis=0)
    new_label = np.asarray(most_common_label).reshape(1, 1)

    rows_to_remove.append(index)
    pseudo_labeled_x = torch.cat((pseudo_labeled_x, torch.from_numpy(new_labeled_sample).to(device)), 0)
    pseudo_labeled_y = torch.cat((pseudo_labeled_y, torch.from_numpy(new_label).to(device)), 0)
    gt_pseudo_labeled.append(y_unlabeled[index])

    if y_unlabeled[index].cpu().detach().numpy() in np.array(known_celltypes):
        should_have_labeled += 1
    
    return pseudo_labeled_x, pseudo_labeled_y, rows_to_remove, gt_pseudo_labeled, should_have_labeled

def print_results(rows_to_remove, sampled_indices, should_have_labeled):
    
    total_labeled_samples = len(rows_to_remove)
    print('Total labeled samples is %d' % total_labeled_samples)
    print('Should have labeled %d from known classes' % len(sampled_indices))
    print('Actually labeled from known classes %d' % should_have_labeled)
    print('Known samples missing %d' % (len(sampled_indices) - should_have_labeled))
    print('Wrongly labeled %d' % (total_labeled_samples - should_have_labeled))



def train_DGI(in_feats, n_hidden, n_layers, dropout, data, device, num_epochs):

    model = DGI(in_feats, n_hidden, n_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if ((epoch % 100) == 0):
            print(f'DGI Epoch [{epoch}/{num_epochs}], Loss: {total_loss:.4f}')

    return model
