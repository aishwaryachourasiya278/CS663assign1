import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def myNearestNeighborInterpolation(img, M, N, M_new, N_new, theta):
    """
    Perform ne  arest neighbor interpolation for rotation, adapting to rotated coordinates.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale or color)
    - M: Original height
    - N: Original width
    - M_new: Output height (same as M for rotation)
    - N_new: Output width (same as N for rotation)
    - theta: Rotation angle in degrees (positive for counterclockwise)
    
    Returns:
    - Rotated image as a NumPy array
    """
    enlarged_img = np.zeros((M_new, N_new), dtype=img.dtype)
    cx = (N - 1) / 2  # Center x
    cy = (M - 1) / 2  # Center y
    theta_rad = math.radians(theta)  # Convert theta to radians
    
    for i in range(M_new):
        for j in range(N_new):
            # Shift to center
            x_shifted = j - cx
            y_shifted = i - cy
            
            # Rotation transformation
            x = x_shifted * math.cos(theta_rad) - y_shifted * math.sin(theta_rad) + cx
            y = x_shifted * math.sin(theta_rad) + y_shifted * math.cos(theta_rad) + cy
            
            # Round to nearest neighbor indices
            x_nearest = round(x)
            y_nearest = round(y)
            
            # Clamp to image boundaries to prevent index errors
            x_nearest = max(0, min(N - 1, x_nearest))
            y_nearest = max(0, min(M - 1, y_nearest))
            
            # Assign nearest neighbor value
            enlarged_img[i, j] = img[y_nearest, x_nearest] if len(img.shape) == 2 else img[y_nearest, x_nearest, :]
    
    return enlarged_img

def myImageRotationUsingNearestNeighborInterp(img, theta):
    """
    Rotate an image using nearest neighbor interpolation around its center, preserving original size.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale or color)
    - theta: Rotation angle in degrees (positive for counterclockwise)
    
    Returns:
    - Rotated image as a NumPy array
    """
    # Get original dimensions and channels
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    # Initialize output image with original size
    rotated = np.zeros_like(img, dtype=img.dtype)
    
    # Perform rotation for each channel if color, or single channel if grayscale
    if channels == 1:
        rotated[:, :] = myNearestNeighborInterpolation(img, height, width, height, width, theta)
    else:
        for c in range(channels):
            rotated[:, :, c] = myNearestNeighborInterpolation(img[:, :, c], height, width, height, width, theta)
    
    return rotated.astype(img.dtype)

# Load the image
image_path = 'data/interp/main.png'
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Convert to floating-point for processing
img_float = img.astype(np.float32)

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

theta = -5  

# Rotate the image
rotated_img = myImageRotationUsingNearestNeighborInterp(img_float, theta)

# Convert rotated image to RGB for display
rotated_rgb = cv2.cvtColor(rotated_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

# Display original and rotated images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rotated_rgb)
plt.title(f'Rotated Image (theta = {theta}°)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the rotated image
cv2.imwrite('Q1/main_rotated_nearest.png', rotated_img)