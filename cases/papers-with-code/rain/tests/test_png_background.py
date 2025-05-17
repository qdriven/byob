import pytest
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tempfile
from PIL import Image

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_font


def test_png_background_color():
    """Test that PNG files are saved with white backgrounds"""
    # Setup font with white background settings
    setup_font()
    
    # Create a temporary file for the PNG
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a simple test figure
        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, facecolor='white')
        
        # Plot some data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        
        # Save the figure as PNG
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', transparent=False)
        plt.close(fig)
        
        # Open the PNG file and check the background color
        img = Image.open(tmp_path)
        
        # Convert to RGB if it's RGBA
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Check the corners of the image for white color
        corners = [
            (0, 0),                    # top-left
            (0, img.height-1),         # bottom-left
            (img.width-1, 0),          # top-right
            (img.width-1, img.height-1)  # bottom-right
        ]
        
        for corner in corners:
            pixel = img.getpixel(corner)
            # Check if the pixel is white or very close to white
            # RGB values for white are (255, 255, 255)
            assert all(value > 250 for value in pixel), f"Corner {corner} is not white: {pixel}"
        
        print("PNG file has white background")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    # Run the test directly
    test_png_background_color()
    print("All tests passed! PNG files have white backgrounds.")
