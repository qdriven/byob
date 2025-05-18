#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layout Parameter Optimizer Module
Neural network model to predict optimal layout parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim

class LayoutParameterOptimizer(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 64, 256, 128], output_size=2):
        """
        Layout parameter optimization neural network
        Args:
            input_size: Input feature dimension
            hidden_sizes: Hidden layer neuron counts
            output_size: Output parameter count (aspect ratio and spacing)
        """
        super(LayoutParameterOptimizer, self).__init__()
        
        # Build four-layer neural network, matching the paper structure
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.output_layer = nn.Linear(hidden_sizes[3], output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward propagation"""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        x = self.output_layer(x)
        
        # Adjust output parameter ranges
        aspect_ratio = 0.5 + 2.0 * self.sigmoid(x[0])  # Aspect ratio range: 0.5-2.5
        spacing = 0.5 + 1.5 * self.sigmoid(x[1])       # Spacing range: 0.5-2.0
        
        return torch.tensor([aspect_ratio, spacing])

class ParameterOptimizerTrainer:
    def __init__(self, model, learning_rate=1e-6):
        """
        Parameter optimizer trainer class
        Args:
            model: Neural network model
            learning_rate: Learning rate
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train_batch(self, features_batch, target_params_batch):
        """
        Train a batch
        Args:
            features_batch: Batch feature data
            target_params_batch: Target parameter data
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        outputs = torch.stack([self.model(x) for x in features_batch])
        loss = self.loss_fn(outputs, target_params_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, path):
        """Save model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model"""
        self.model.load_state_dict(torch.load(path))