import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def myBilinearInterpolation(img, new_height, new_width, theta):
    """
    Perform bilinear interpolation for rotation, adapting to rotated coordinates.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale or color)
    - new_height: Target number of rows (same as original height for rotation)
    - new_width: Target number of columns (same as original width for rotation)
    - theta: Rotation angle in degrees (positive for counterclockwise)
    
    Returns:
    - Rotated image as a NumPy array (floating-point)
    """
    resized = np.zeros((new_height, new_width), dtype=np.float32)
    height, width = img.shape[:2]
    cx = (width - 1) / 2  # Center x
    cy = (height - 1) / 2  # Center y
    theta_rad = math.radians(theta)  # Convert theta to radians
    
    for y in range(new_height):
        for x in range(new_width):
            # Shift to center
            x_shifted = x - cx
            y_shifted = y - cy
            
            # Rotation transformation to map output to input coordinates
            orig_x = x_shifted * math.cos(theta_rad) - y_shifted * math.sin(theta_rad) + cx
            orig_y = x_shifted * math.sin(theta_rad) + y_shifted * math.cos(theta_rad) + cy
            
            # Adapt bilinear interpolation to rotated coordinates
            if 0 <= orig_y < height - 1 and 0 <= orig_x < width - 1:
                x1 = np.floor(orig_x).astype(int)
                x2 = np.ceil(orig_x).astype(int)
                y1 = np.floor(orig_y).astype(int)
                y2 = np.ceil(orig_y).astype(int)
                
                t1 = orig_x - x1
                A = img[y1, x1] if len(img.shape) == 2 else img[y1, x1, 0]
                C = img[y2, x1] if len(img.shape) == 2 else img[y2, x1, 0]
                B = img[y1, x2] if len(img.shape) == 2 else img[y1, x2, 0]
                D = img[y2, x2] if len(img.shape) == 2 else img[y2, x2, 0]
                
                y1_val = (1 - t1) * A + t1 * B
                y2_val = (1 - t1) * C + t1 * D
                
                t2 = orig_y - y1
                t3 = (1 - t2) * y1_val + t2 * y2_val
                
                resized[y, x] = t3
            else:
                resized[y, x] = 0.0  # Black for out-of-bounds
    
    return resized

def myImageRotationUsingBilinearInterp(img, theta):
    """
    Rotate an image using bilinear interpolation around its center, preserving original size.
    
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
    rotated = np.zeros_like(img, dtype=np.float32)
    
    # Perform rotation for each channel if color, or single channel if grayscale
    if channels == 1:
        rotated[:, :] = myBilinearInterpolation(img, height, width, theta)
    else:
        for c in range(channels):
            rotated[:, :, c] = myBilinearInterpolation(img[:, :, c], height, width, theta)
    
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
rotated_img = myImageRotationUsingBilinearInterp(img_float, theta)

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
cv2.imwrite('Q1/main_rotated_bilinear.png', rotated_img)