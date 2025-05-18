#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessor Module
Handles connection matrix creation and feature extraction using SVD
"""

import numpy as np
import torch
from scipy.linalg import svd

class DataPreprocessor:
    def __init__(self, svd_components=40):
        """
        Initialize data preprocessor
        Args:
            svd_components: Number of SVD components to keep
        """
        self.svd_components = svd_components
    
    def create_connection_matrix(self, connections):
        """
        Create connection relationship matrix
        Args:
            connections: List of tuples (component1, component2) representing connections
        Returns:
            Connection matrix and component-to-index mapping
        """
        # Get all unique component IDs
        all_components = set()
        for comp1, comp2 in connections:
            all_components.add(comp1)
            all_components.add(comp2)
        
        # Create ID to index mapping
        comp_to_idx = {comp: idx for idx, comp in enumerate(all_components)}
        n = len(all_components)
        
        # Create connection matrix
        conn_matrix = np.zeros((n, n))
        for comp1, comp2 in connections:
            i, j = comp_to_idx[comp1], comp_to_idx[comp2]
            conn_matrix[i, j] = 1
            conn_matrix[j, i] = 1  # Symmetric matrix
        
        return conn_matrix, comp_to_idx
    
    def compress_connection_matrix(self, conn_matrix):
        """
        Use SVD to compress connection matrix and extract features
        Args:
            conn_matrix: Connection relationship matrix
        Returns:
            Compressed feature vector
        """
        # Perform SVD decomposition
        U, s, Vt = svd(conn_matrix, full_matrices=False)
        
        # Take the first k singular values and corresponding features
        k = min(self.svd_components, len(s))
        features = U[:, :k] @ np.diag(s[:k])
        
        # Flatten feature vector to 1D
        flattened_features = features.flatten()
        
        return torch.tensor(flattened_features, dtype=torch.float32)