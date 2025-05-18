#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ship Electrical Layout System
Main entry point for the AI-based ship electrical layout system
"""

import argparse
import os
import sys
import torch

from data_preprocessor import DataPreprocessor
from layout_optimizer import LayoutParameterOptimizer, ParameterOptimizerTrainer
from layout_engine import PriorRuleLayoutEngine, Component, SubGraph
from visualizer import LayoutVisualizer

def main():
    parser = argparse.ArgumentParser(description='Ship Electrical Layout System')
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'],
                        help='Operation mode: train or inference')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--model', type=str, default='model.pth', help='Model path')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the system
    system = ShipElectricalLayoutSystem(model_path=args.model if os.path.exists(args.model) else None)
    
    if args.mode == 'train':
        # Load training data
        training_data = load_training_data(args.input)
        # Train the model
        system.train(training_data, epochs=100)
        # Save the model
        system.save_model(os.path.join(args.output, 'model.pth'))
    else:
        # Load components and connections data
        components_data, connections_data = load_inference_data(args.input)
        # Process layout
        layout, wiring = system.process_layout(components_data, connections_data)
        # Visualize and save results
        visualizer = LayoutVisualizer(layout, wiring, components_data)
        visualizer.visualize(save_path=os.path.join(args.output, 'layout.png'))
        visualizer.export_to_cad(os.path.join(args.output, 'layout.dxf'))

def load_training_data(file_path):
    """Load training data from file"""
    # Implement data loading logic here
    # For now, return dummy data
    return []

def load_inference_data(file_path):
    """Load inference data from file"""
    # Implement data loading logic here
    # For now, return dummy data
    components_data = [
        {'id': 'comp1', 'name': 'Component 1'},
        {'id': 'comp2', 'name': 'Component 2'},
        {'id': 'comp3', 'name': 'Component 3'},
    ]
    connections_data = [
        ('comp1', 'comp2'),
        ('comp2', 'comp3'),
    ]
    return components_data, connections_data

class ShipElectricalLayoutSystem:
    """Ship Electrical Layout System"""
    def __init__(self, model_path=None):
        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(svd_components=40)
        
        # Initialize neural network model
        input_size = 40 * 40  # SVD compressed feature dimension
        self.model = LayoutParameterOptimizer(input_size)
        
        # Load pre-trained model if available
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        # Initialize layout engine
        self.layout_engine = PriorRuleLayoutEngine()
    
    def train(self, training_data, epochs=100, batch_size=64, learning_rate=1e-6):
        """Train the neural network model"""
        trainer = ParameterOptimizerTrainer(self.model, learning_rate)
        
        # Prepare training data
        features = []
        targets = []
        
        for connections, target_params in training_data:
            # Generate connection matrix
            conn_matrix, _ = self.preprocessor.create_connection_matrix(connections)
            # Compress features
            feature = self.preprocessor.compress_connection_matrix(conn_matrix)
            features.append(feature)
            targets.append(torch.tensor(target_params, dtype=torch.float32))
        
        # Train model
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(features))
            
            # Batch training
            total_loss = 0
            for i in range(0, len(features), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = [features[idx] for idx in batch_indices]
                batch_targets = torch.stack([targets[idx] for idx in batch_indices])
                
                loss = trainer.train_batch(batch_features, batch_targets)
                total_loss += loss
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(features):.6f}")
        
        return trainer
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
    
    def process_layout(self, components_data, connections_data, root_id=None):
        """Process layout"""
        # Load data
        self.layout_engine.load_components(components_data)
        self.layout_engine.load_connections(connections_data)
        
        # If root node is not specified, select the first component
        if root_id is None and components_data:
            root_id = components_data[0]['id']
        
        # Generate connection matrix
        conn_matrix, _ = self.preprocessor.create_connection_matrix(connections_data)
        
        # Compress features
        feature = self.preprocessor.compress_connection_matrix(conn_matrix)
        
        # Use neural network to predict layout parameters
        with torch.no_grad():
            layout_params = self.model(feature)
        
        # Use prior rules to generate layout
        layout = self.layout_engine.generate_layout(root_id, layout_params)
        
        # Generate wiring
        wiring = self.layout_engine.generate_wiring(layout)
        
        return layout, wiring

if __name__ == "__main__":
    main()