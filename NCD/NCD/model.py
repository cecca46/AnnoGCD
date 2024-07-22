import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number
from torch_geometric.nn import GCNConv


class MaskedAutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=[32, 32], dropout=0.2, mask_prob=0.2):
        super().__init__()
        self.input_size = input_size
        self. mask_prob = torch.tensor(mask_prob)

        self.encoder, self.mask_predictor, self.decoder = self.contrsuct_model(input_size, hidden_size, dropout)
        
    def contrsuct_model(self, input_size, hidden_size, dropout):
        
        # Dynamically create the encoder layers
        layers = [nn.Dropout(p=dropout)]
        input_size = input_size
                
        if isinstance(hidden_size, Number):
            hidden_size = [hidden_size]
            
        hidden_size.insert(0, input_size)
        
        for idx in range(len(hidden_size)-2):
            layers.extend([
                nn.Linear(hidden_size[idx], hidden_size[idx+1]),
                nn.LayerNorm(hidden_size[idx+1]),
                nn.Mish(inplace=True)
            ])
        
        # Last layer of the encoder without activation
        layers.append(nn.Linear(hidden_size[-2], hidden_size[-1]))

        encoder = nn.Sequential(*layers)

        mask_predictor = nn.Linear(hidden_size[-1], input_size)
        decoder = nn.Linear(in_features=hidden_size[-1] + input_size, out_features=input_size)
        
        return encoder, mask_predictor, decoder

    def forward(self, X):
        
        if self.training:
            corrupted_X, true_mask = self.mask_input(X)
        else:
            corrupted_X = X
            true_mask = torch.ones_like(X)
        
        encoded = self.encoder(corrupted_X)
        predicted_mask = self.mask_predictor(encoded)
        masked_encoded = torch.cat([encoded, true_mask * corrupted_X], dim=-1)
        reconstruction = self.decoder(masked_encoded)
        
        return [encoded, reconstruction]
    
    
    def mask_input(self, X):
        
        should_swap = torch.bernoulli(self.mask_prob.to( X.device) * torch.ones((X.shape)).to(X.device))
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        masked = (corrupted_X != X).float()
        return corrupted_X, masked


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.predictor(x)
    
class MultiObjectiveLoss(nn.Module):
    def __init__(self):
        super(MultiObjectiveLoss, self).__init__()
        self.weights = nn.Parameter(torch.rand(2)) 
    
    def forward(self, loss1, loss2):
        # Compute the weighted sum of the loss terms
        weights = F.softmax(self.weights, dim=0)
        weighted_loss = self.weights[0] * loss1 + self.weights[1] * loss2
        return weighted_loss
    

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden))

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = F.relu(layer(x, edge_index))
        return x

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN(in_feats, n_hidden, n_layers, dropout)

    def forward(self, data, corrupt=False):
        x, edge_index = data.x, data.edge_index
        if corrupt:
            perm = torch.randperm(x.size(0))
            x = x[perm]
        x = self.conv(x, edge_index)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / torch.sqrt(torch.tensor(size, dtype=torch.float))
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features

class DGI(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, n_layers, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, data):
        positive = self.encoder(data, corrupt=False)
        negative = self.encoder(data, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2