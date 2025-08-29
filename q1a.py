import cv2
import numpy as np
import matplotlib.pyplot as plt

def myImageShrink(img, d):
    """
    Shrink an image by subsampling every d-th pixel along rows and columns.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale or color)
    - d: Subsampling factor (integer, e.g., 2 or 3)
    
    Returns:
    - Subsampled image as a NumPy array (floating-point)
    """
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate new dimensions
    new_height = height // d
    new_width = width // d
    
    # Subsample by selecting every d-th pixel
    if len(img.shape) == 3:  # Color image (RGB)
        subsampled = img[ : :d, : :d, :]  #img[start:end:step, start:end:step, :]
    else:  # Grayscale image
        subsampled = img[ : :d, : :d]   
    
    return subsampled

# Load the image
image_path = 'data/interp/suit.png'
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

print(img[11][14])
print(f"Image shape is : {img.shape}")

# Convert to floating-point (np.float32) to avoid integer rounding
img = img.astype(np.float32)

# Handle color channels
channels = img.shape[2] if len(img.shape) == 3 else 1
if channels == 4:  # RGBA, extract RGB channels
    img = img[:, :, :3] #channels 0,1 and 2 extracted
    print("Note: RGBA image detected, using RGB channels only.")
elif channels == 3:  # BGR (OpenCV default), convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("RGB image detected / BGR converted to RGB")
else:  # Grayscale
    print("Note: Grayscale image detected.")

# Get original dimensions
height, width = img.shape[:2]
print(f"Original Dimensions: {width} width x{height} height")

# Perform subsampling for d = 2 and d = 3
img_d2 = myImageShrink(img, d=2)
img_d3 = myImageShrink(img, d=3)


cmap = 'jet'  
norm = plt.Normalize(vmin=0, vmax=255)  # Normalize to 0-255 for float32 pixel values

# Create a figure for visualization
plt.figure(figsize=(18, 5))

# Display original image
plt.subplot(1, 3, 1)
if channels == 1:
    im = plt.imshow(img, cmap=cmap, norm=norm)
else:
    im = plt.imshow(img / 255.0, cmap=cmap, norm=norm)  # Scale to [0,1] for RGB display
plt.title(f'Original\n{width}x{height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.colorbar(im, label='Intensity' if channels == 1 else 'RGB Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

# Display subsampled image (d=2)
height_d2, width_d2 = img_d2.shape[:2]
plt.subplot(1, 3, 2)
if channels == 1:
    im = plt.imshow(img_d2, cmap=cmap, norm=norm)
else:
    im = plt.imshow(img_d2 / 255.0, cmap=cmap, norm=norm)
plt.title(f'Subsampled (d=2)\n{width_d2}x{height_d2}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.colorbar(im, label='Intensity' if channels == 1 else 'RGB Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

# Display subsampled image (d=3)
height_d3, width_d3 = img_d3.shape[:2]
plt.subplot(1, 3, 3)
if channels == 1:
    im = plt.imshow(img_d3, cmap=cmap, norm=norm)
else:
    im = plt.imshow(img_d3 / 255.0, cmap=cmap, norm=norm)
plt.title(f'Subsampled (d=3)\n{width_d3}x{height_d3}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.colorbar(im, label='Intensity' if channels == 1 else 'RGB Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save subsampled images (optional, in floating-point)
cv2.imwrite('Q1/suit_d2.png', cv2.cvtColor(img_d2, cv2.COLOR_RGB2BGR) if channels == 3 else img_d2)
cv2.imwrite('Q1/suit_d3.png', cv2.cvtColor(img_d3, cv2.COLOR_RGB2BGR) if channels == 3 else img_d3)
