#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prior Rule Layout Engine Module
Implements layout rules based on connection relationships
"""

import numpy as np

class Component:
    """Component class"""
    def __init__(self, id, name, width=1, height=1, position=(0, 0)):
        self.id = id
        self.name = name
        self.width = width
        self.height = height
        self.position = position
        self.connections = []  # Store connected component IDs

class SubGraph:
    """SubGraph class, representing a group of laid out components"""
    def __init__(self, components=None, width=0, height=0, position=(0, 0)):
        self.components = components or []
        self.width = width
        self.height = height
        self.position = position
        self.connections = []  # Store connected component or subgraph IDs

class PriorRuleLayoutEngine:
    """Prior rule layout engine"""
    def __init__(self):
        self.components = {}  # Store all components
        self.subgraphs = {}   # Store all subgraphs
        self.connections = [] # Store all connection relationships
    
    def load_components(self, components_data):
        """Load component data"""
        for comp_data in components_data:
            comp = Component(
                id=comp_data['id'],
                name=comp_data['name'],
                width=comp_data.get('width', 1),
                height=comp_data.get('height', 1)
            )
            self.components[comp.id] = comp
    
    def load_connections(self, connections_data):
        """Load connection relationship data"""
        self.connections = connections_data
        # Update component connection information
        for comp1_id, comp2_id in connections_data:
            if comp1_id in self.components:
                self.components[comp1_id].connections.append(comp2_id)
            if comp2_id in self.components:
                self.components[comp2_id].connections.append(comp1_id)
    
    def layout_component_series(self, component_ids, aspect_ratio, spacing):
        """
        Component series scenario layout
        Args:
            component_ids: List of component IDs to layout
            aspect_ratio: Aspect ratio
            spacing: Spacing
        Returns:
            Completed subgraph
        """
        n = len(component_ids)
        
        # Calculate rows and columns
        if aspect_ratio >= 1:
            cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
            rows = int(np.ceil(n / cols))
        else:
            rows = int(np.ceil(np.sqrt(n / aspect_ratio)))
            cols = int(np.ceil(n / rows))
        
        # Optimize rows and columns
        if (rows - 1) * cols >= n:
            rows -= 1
        if (cols - 1) * rows >= n:
            cols -= 1
        
        # Layout components (S-shaped arrangement)
        positions = {}
        for i, comp_id in enumerate(component_ids):
            row = i // cols
            col = i % cols if row % 2 == 0 else cols - 1 - (i % cols)  # S-shaped arrangement
            positions[comp_id] = (col * (1 + spacing), row * (1 + spacing))
        
        # Create subgraph
        width = cols * (1 + spacing) - spacing
        height = rows * (1 + spacing) - spacing
        
        # Update component positions
        for comp_id, pos in positions.items():
            self.components[comp_id].position = pos
        
        # Create and return subgraph
        subgraph = SubGraph(
            components=[self.components[cid] for cid in component_ids],
            width=width,
            height=height
        )
        
        return subgraph
    
    def layout_component_subgraph_series(self, subgraph_id, component_ids, aspect_ratio, spacing):
        """
        Component-subgraph series scenario layout
        Args:
            subgraph_id: Subgraph ID
            component_ids: List of component IDs to layout
            aspect_ratio: Aspect ratio
            spacing: Spacing
        Returns:
            Completed subgraph
        """
        subgraph = self.subgraphs[subgraph_id]
        
        # Subgraph at the top
        subgraph_width = subgraph.width
        
        # Calculate component layout
        n = len(component_ids)
        cols = max(1, int(subgraph_width / (1 + spacing)))
        rows = int(np.ceil(n / cols))
        
        # Layout components (vertical S-shape)
        positions = {}
        for i, comp_id in enumerate(component_ids):
            row = i // cols
            col = i % cols if row % 2 == 0 else cols - 1 - (i % cols)
            positions[comp_id] = (col * (1 + spacing), subgraph.height + (1 + spacing) + row * (1 + spacing))
        
        # Update component positions
        for comp_id, pos in positions.items():
            self.components[comp_id].position = pos
        
        # Calculate new subgraph dimensions
        width = max(subgraph_width, cols * (1 + spacing) - spacing)
        height = subgraph.height + (1 + spacing) + rows * (1 + spacing) - spacing
        
        # Create and return subgraph
        new_components = subgraph.components.copy()
        new_components.extend([self.components[cid] for cid in component_ids])
        
        new_subgraph = SubGraph(
            components=new_components,
            width=width,
            height=height
        )
        
        return new_subgraph
    
    def layout_component_subgraph_parallel(self, items, parent_id, aspect_ratio, spacing):
        """
        Component-subgraph parallel scenario layout
        Args:
            items: List of items to layout, format [(id, is_subgraph)]
            parent_id: Parent node ID
            aspect_ratio: Aspect ratio
            spacing: Spacing
        Returns:
            Completed subgraph
        """
        # Sort by size (large to small)
        sorted_items = []
        for item_id, is_subgraph in items:
            if is_subgraph:
                size = self.subgraphs[item_id].width * self.subgraphs[item_id].height
                sorted_items.append((item_id, is_subgraph, size))
            else:
                sorted_items.append((item_id, is_subgraph, 1))  # Default component size is 1
        
        sorted_items.sort(key=lambda x: x[2], reverse=True)
        
        # Layout from left to right
        current_x = 0
        max_height = 0
        
        for item_id, is_subgraph, _ in sorted_items:
            if is_subgraph:
                subgraph = self.subgraphs[item_id]
                # Update subgraph position
                subgraph.position = (current_x, 0)
                current_x += subgraph.width + spacing
                max_height = max(max_height, subgraph.height)
            else:
                # Update component position
                self.components[item_id].position = (current_x, 0)
                current_x += 1 + spacing
                max_height = max(max_height, 1)
        
        # Place parent node at bottom center
        parent_x = (current_x - spacing) / 2 - 0.5
        parent_y = max_height + spacing + 1
        self.components[parent_id].position = (parent_x, parent