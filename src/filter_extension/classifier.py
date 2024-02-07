from torch import nn


# this is the initialization line in the original code:
# models[m] = getattr(model_list, args.models[m].model)()
class Classifier(nn.Module):
    def __init__(self, dim_input, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_input, 780),
            nn.ReLU(),
            nn.Linear(780, 360),
            nn.ReLU(),
            nn.Linear(360, num_classes),
        )
        # self.softmax = nn.Softmax(dim=1)

    # should return logits and features
    def forward(self, x):
        logits = self.model(x)
        # prob = self.softmax(logits)
        return logits


'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, dim_input, num_channels, hidden_units, num_classes):
        super(CNN1D, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=hidden_units, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2, padding=1)

        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate length of the features after the conv and pooling layers
        # Adjust this if you add more layers or change parameters
        self.feature_length = hidden_units * (dim_input // 2 // 2)

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_length, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        # Apply conv + relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, self.feature_length)

        # Apply fully connected layers with relu activation for fc1
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, sequence_length, num_classes, n_features=1, n_filters=64, kernel_size=3, pool_size=2):
        super(CNN1D, self).__init__()

        # Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=kernel_size)
        # Max Pooling Layer
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        # Calculate the number of output features after Conv and Pooling
        conv_out_size = self._calculate_conv_output_size(sequence_length, kernel_size, pool_size)
        # Fully connected layer
        self.fc1 = nn.Linear(n_filters * conv_out_size, 50)
        # Output layer
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # Convolution and activation function
        x = F.relu(self.conv1(x))
        # Pooling
        x = self.pool1(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x

    def _calculate_conv_output_size(self, L_in, kernel_size, pool_size):
        # Formula for calculating the size of the output of a convolutional layer
        L_out = (L_in - kernel_size + 1) // pool_size
        return L_out

'''
import numpy as np
import torch

class ReductionModel(nn.Module):
    def __init__(self, num_classes, dim_input_single,device):
        super().__init__()
        self.reduction = [nn.Sequential(
            nn.Linear(dim_input_single, 250),
            nn.ReLU(),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Linear(125, 30))
            for _ in range(num_classes)]
        self.device = device
        self.final_layer = nn.Sequential(
            nn.Linear(30*num_classes, 15*num_classes),
            nn.ReLU(),
            nn.Linear(15*num_classes, 5*num_classes),
            nn.ReLU(),
            nn.Linear(5*num_classes, num_classes),
        ).to(device)

        # Move each reduction block to the device
        for block in self.reduction:
            block.to(device)

    def forward(self, x):
        dimred = []
        # resizing
        x = x.to(self.device)
        #x_res = x.reshape(x.shape[0], 1, x.shape[1]).to(self.device)
        #x_res = np.dsplit(x_res, 25)  # divido in 25 vettori
        #for idx, part in enumerate(x_res):
        #    dimred.append(self.reduction[idx](part))
        #return

        splits = x.split(x.shape[1] // 25, dim=1)  # Split along the feature dimension
        dimred = [self.reduction[idx](split) for idx, split in enumerate(splits)]

        # Concatenate the outputs from each reduction block
        concatenated = torch.cat(dimred, dim=1)

        # Pass the concatenated result through the final layer
        return self.final_layer(concatenated)

