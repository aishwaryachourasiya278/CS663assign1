import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def adaptive_threshold(img, block_size=15, c=10):
    """
    Perform adaptive thresholding using mean of local window.
    Pixels darker than mean - c in their block are set black; else white.

    Args:
        img (numpy.ndarray): Input grayscale image (2D NumPy array).
        block_size (int): Odd integer, size of local window.
        c (int): Constant subtracted from mean.

    Returns:
        numpy.ndarray: Binarized image (2D NumPy array of 0/255).
    """
    half_window = block_size // 2
    padded = np.pad(img, half_window, mode='reflect')
    out = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Get local window
            window = padded[i:i + block_size, j:j + block_size]
            local_mean = np.mean(window)
            # Adaptive thresholding
            out[i, j] = 0 if img[i, j] < local_mean - c else 255
    return out

# Image paths and parameters
image_paths = [
    'data/thresh/receipt.png',
    'data/thresh/blackboard.png',
    'data/thresh/lilavati.tif',
    'data/thresh/qr.png'
]
params = [
    {'block_size': 15, 'c': 10}, #15 #10
    {'block_size': 29, 'c': 15}, #29 #15
    {'block_size': 31, 'c': 8}, #31 #8
    {'block_size': 11, 'c': 2} #11 #2
]

if __name__ == "__main__":
    # List to store figure handles
    figures = []
    
    # Process each image in a separate window
    for i, (path, param) in enumerate(zip(image_paths, params)):
        # Load image
        img = Image.open(path).convert('L')
        img_np = np.array(img)
        
        # Apply adaptive thresholding
        th = adaptive_threshold(img_np, block_size=param['block_size'], c=param['c'])
        
        # Create a new figure for each image
        fig = plt.figure(figsize=(12, 5))
        figures.append(fig)
        fig.suptitle(f'Image {i + 1}')
        
        # Display original grayscale
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        
        # Display thresholded image
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(th, cmap='gray')
        ax2.set_title('Thresholded')
        ax2.axis('off')
        
        plt.tight_layout()
    
    # Show all figures and block until closed
    plt.show()