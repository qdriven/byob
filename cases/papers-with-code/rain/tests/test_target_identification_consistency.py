import pytest
import numpy as np
from datetime import datetime
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process import read_observation_data, interpolate_to_grid, read_model_data
from target_visualization import identify_targets
from visualization import identify_and_plot_obs_clusters


class TestTargetIdentificationConsistency:
    """Test cases for consistency between target identification functions"""

    @pytest.fixture(scope="class")
    def test_data(self):
        """Set up test data for all test methods"""
        # Load observation data
        daily_rainfall = read_observation_data('观测数据202107.txt')
        grid_data = interpolate_to_grid(daily_rainfall)
        
        # Load model data
        model_data = read_model_data('模式数据/')
        
        # Test dates
        test_dates = [
            datetime(2021, 7, 18),  # Main test date
            datetime(2021, 7, 19),  # Additional test date
            datetime(2021, 7, 20)   # Additional test date
        ]
        
        return {
            'grid_data': grid_data,
            'model_data': model_data,
            'test_dates': test_dates
        }

    def test_functions_produce_same_number_of_targets(self, test_data):
        """Test that identify_targets and identify_and_plot_obs_clusters produce the same number of targets"""
        grid_data = test_data['grid_data']
        
        for date in test_data['test_dates']:
            if date not in grid_data:
                continue
                
            obs_data = grid_data[date]
            
            # Run both functions with the same parameters
            targets = identify_targets(obs_data, threshold=25, min_size=8, separate=True)
            clusters = identify_and_plot_obs_clusters({date: obs_data}, threshold=25, min_size=8)
            
            # Check that the number of targets/clusters is the same
            assert len(targets) == len(clusters[date]), \
                f"Functions produced different number of targets/clusters for {date}"
            
            # Print the results for debugging
            print(f"\nDate: {date}")
            print(f"Number of targets from identify_targets: {len(targets)}")
            print(f"Number of clusters from identify_and_plot_obs_clusters: {len(clusters[date])}")

    def test_different_thresholds(self, test_data):
        """Test both functions with different threshold values"""
        grid_data = test_data['grid_data']
        date = test_data['test_dates'][0]  # Use the first test date
        
        if date not in grid_data:
            pytest.skip(f"Test date {date} not in grid data")
            
        obs_data = grid_data[date]
        
        for threshold in [10, 25, 50]:
            # Run both functions with the same parameters
            targets = identify_targets(obs_data, threshold=threshold, min_size=8, separate=True)
            clusters = identify_and_plot_obs_clusters({date: obs_data}, threshold=threshold, min_size=8)
            
            # Check that the number of targets/clusters is the same
            assert len(targets) == len(clusters[date]), \
                f"Functions produced different number of targets/clusters for threshold={threshold}"
            
            # Print the results for debugging
            print(f"\nThreshold: {threshold}")
            print(f"Number of targets from identify_targets: {len(targets)}")
            print(f"Number of clusters from identify_and_plot_obs_clusters: {len(clusters[date])}")

    def test_different_min_sizes(self, test_data):
        """Test both functions with different min_size values"""
        grid_data = test_data['grid_data']
        date = test_data['test_dates'][0]  # Use the first test date
        
        if date not in grid_data:
            pytest.skip(f"Test date {date} not in grid data")
            
        obs_data = grid_data[date]
        
        for min_size in [5, 8, 10]:
            # Run both functions with the same parameters
            targets = identify_targets(obs_data, threshold=25, min_size=min_size, separate=True)
            clusters = identify_and_plot_obs_clusters({date: obs_data}, threshold=25, min_size=min_size)
            
            # Check that the number of targets/clusters is the same
            assert len(targets) == len(clusters[date]), \
                f"Functions produced different number of targets/clusters for min_size={min_size}"
            
            # Print the results for debugging
            print(f"\nMin size: {min_size}")
            print(f"Number of targets from identify_targets: {len(targets)}")
            print(f"Number of clusters from identify_and_plot_obs_clusters: {len(clusters[date])}")
