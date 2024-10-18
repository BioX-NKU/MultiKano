import scipy.sparse
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
import scanpy as sc
import episcanpy as epi
import torch.nn.functional as F
import math
import numpy as np
from .utils import *
from .data_processing import *


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=4, spline_order=2, scale_noise=0.01, scale_base=0.5,
                 scale_spline=0.5, enable_standalone_scale_spline=True, base_activation=nn.LeakyReLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        spline_output = nn.functional.linear(self.b_splines(x).view(x.size(0), -1),
                                             self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=4, spline_order=2, scale_noise=0.01, scale_base=0.5, scale_spline=0.5,
                 base_activation=nn.LeakyReLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                         scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                         base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiOmicsNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiOmicsNet, self).__init__()
        self.fc1 = KAN([input_size, 128, hidden_size])
        self.fc2 = KAN([hidden_size, 128, hidden_size])
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.batch_norm(self.fc1(x))
        x = self.activation(x)
        x = self.dropout(self.fc2(x))
        x_out = self.fc3(x)
        return x_out


class CustomLoss(nn.Module):
    def __init__(self, cross_entropy_weight=1.0, cosine_similarity_weight=0.1):
        super(CustomLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.cosine_similarity_weight = cosine_similarity_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, first_round_features, second_round_features):
        loss_ce = self.cross_entropy_loss(outputs, labels)
        predicted_cos_similarities = F.cosine_similarity(first_round_features, second_round_features, dim=1).view(-1, 1)
        loss_cos = F.mse_loss(predicted_cos_similarities, torch.ones_like(predicted_cos_similarities))
        total_loss = self.cross_entropy_weight * loss_ce + self.cosine_similarity_weight * loss_cos
        return total_loss

def run_model(train_data_rna, train_data_atac, test_data_rna,  test_data_atac):

    train_features, train_labels, test_features, test_labels, first_round_features, second_round_features, num_classes = data_processing(
    train_data_rna, test_data_rna, train_data_atac, test_data_atac)
    input_size = train_features.shape[1]
    hidden_size = 256
    num_epochs = 150
    initial_lr = 0.001
    T_max = 50

    model = MultiOmicsNet(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    criterion = CustomLoss(cross_entropy_weight=1.0, cosine_similarity_weight=0.1)



    model = MultiOmicsNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    criterion = CustomLoss(cross_entropy_weight=1.0, cosine_similarity_weight=0.1)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features_tensor)
        loss = criterion(outputs, labels_train_tensor, train_features_tensor[:len(train_labels)],
                         train_features_tensor[len(train_labels):])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        outputs = model(test_features_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.numpy()