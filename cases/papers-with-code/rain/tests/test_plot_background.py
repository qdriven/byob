import pytest
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_font
from target_visualization import plot_targets_with_markers


def test_plot_background_color():
    """Test that plots have a white background"""
    # Setup font with white background settings
    setup_font()
    
    # Create a simple test figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Plot some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    
    # Check that the figure and axes have white backgrounds
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0), "Figure background is not white"
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0), "Axes background is not white"
    
    # Clean up
    plt.close(fig)


def test_cartopy_plot_background_color():
    """Test that cartopy plots have a white background"""
    try:
        import cartopy.crs as ccrs
    except ImportError:
        pytest.skip("Cartopy not installed, skipping test")
    
    # Setup font with white background settings
    setup_font()
    
    # Create a simple test figure with cartopy projection
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Check that the figure and axes have white backgrounds
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0), "Figure background is not white"
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0), "Axes background is not white"
    
    # Clean up
    plt.close(fig)


if __name__ == "__main__":
    # Run the tests directly
    test_plot_background_color()
    test_cartopy_plot_background_color()
    print("All tests passed! Plot backgrounds are white.")
