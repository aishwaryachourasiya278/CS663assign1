import cv2
import numpy as np
import matplotlib.pyplot as plt


def myBilinearInterpolation(img, new_height, new_width):
    """
    Resize image using bilinear interpolation, ensuring first/last rows/columns match.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale)
    - new_height: Target number of rows (300*(M-1)+1)
    - new_width: Target number of columns (300*(N-1)+1)
    
    Returns:
    - Resized image as a NumPy array (floating-point)
    """
    resized = np.zeros((new_height,new_width))
    for y in range(new_height):
        for x in range(new_width):
            A = img[np.floor(y/300).astype(int)][np.floor(x/300).astype(int)]
            C = img[np.ceil(y/300).astype(int)][np.floor(x/300).astype(int)]
            B = img[np.floor(y/300).astype(int)][np.ceil(x/300).astype(int)]
            D = img[np.ceil(y/300).astype(int)][np.ceil(x/300).astype(int)]   # Qxy acc to slide
            
            t1 = ( x/300 - np.floor(x/300) )#( np.ceil(x/300) - np.floor(x/300) )
            y1 = (1-t1)*A + t1*B
            y2 = (1-t1)*C + t1*D

            t2 = ( y/300 - np.floor(y/300) )#/( np.ceil(y/300) - np.floor(y/300) )
            
            t3 = (1-t2)*y1 + t2*y2

            resized[y][x] = t3
    return resized


    '''GENERATED AND CORRECT 
    height, width = img.shape[:2]
    resized = np.zeros((new_height, new_width), dtype=np.float32)
    
    # Create coordinate grids for output image
    y, x = np.indices((new_height, new_width), dtype=np.float32)
    
    # Map output coordinates to input coordinates
    src_x = x * (width - 1) / (new_width - 1)  # 0 to 3
    src_y = y * (height - 1) / (new_height - 1)  # 0 to 3
    
    # Get integer coordinates of four neighboring pixels
    x_floor = np.floor(src_x).astype(int)
    x_ceil = np.ceil(src_x).astype(int)
    y_floor = np.floor(src_y).astype(int)
    y_ceil = np.ceil(src_y).astype(int)
    
    # Clip to ensure valid indices
    x_floor = np.clip(x_floor, 0, width - 1)
    x_ceil = np.clip(x_ceil, 0, width - 1)
    y_floor = np.clip(y_floor, 0, height - 1)
    y_ceil = np.clip(y_ceil, 0, height - 1)
    
    # Compute fractional distances
    dx = src_x - x_floor
    dy = src_y - y_floor
    
    # Handle boundaries (where x_floor = x_ceil or y_floor = y_ceil)
    dx[x_floor == x_ceil] = 0
    dy[y_floor == y_ceil] = 0
    
    # Get pixel values (Qxy notation from slides, corrected)
    q22 = img[y_floor, x_floor]  # (x1, y1)
    q12 = img[y_ceil, x_floor]   # (x1, y2)
    q21 = img[y_floor, x_ceil]   # (x2, y1)
    q11 = img[y_ceil, x_ceil]    # (x2, y2)
    
    # Compute bilinear interpolation weights (Y, V, G, R from slides, corrected)
    Y = q22 * (1 - dx) * (1 - dy)  # (x1, y1)
    V = q12 * (1 - dx) * dy        # (x1, y2)
    G = q21 * dx * (1 - dy)        # (x2, y1)
    R = q11 * dx * dy              # (x2, y2)
    
    resized = Y + V + G + R
    
    return resized'''


# Load image (assuming 4x4 grayscale for testing)
image_path = 'data/interp/random.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Convert to floating-point
img = img.astype(np.float32)

# Verify dimensions (expect 4x4)
height, width = img.shape[:2]

# Compute new dimensions
new_height = 300 * (height - 1) + 1  # 300*(4-1)+1 = 901
new_width = 300 * (width - 1) + 1   # 300*(4-1)+1 = 901
print(f"Original Dimensions: {width}x{height}")
print(f"Resized Dimensions: {new_width}x{new_height}")

# Resize using nearest-neighbor interpolation
resized_img = myBilinearInterpolation(img, new_height, new_width)

# Set colormap (jet, 256 colors)
cmap = 'jet'
norm = plt.Normalize(vmin=0, vmax=255)

# Create figure
plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
im = plt.imshow(img, cmap=cmap, norm=norm, extent=(0, width, height, 0))
plt.title(f'Original\n{width}x{height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.xticks(np.arange(0, width, 1))  # Integer ticks: 0, 1, 2, 3
plt.yticks(np.arange(0, height, 1))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

# Resized image
plt.subplot(1, 2, 2)
im = plt.imshow(resized_img, cmap=cmap, norm=norm, extent=(0, new_width, new_height, 0))
plt.title(f'Resized\n{new_width}x{new_height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.xticks(np.arange(0, new_width, 100))  # Integer ticks every 100 for readability
plt.yticks(np.arange(0, new_height, 100))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Save resized image (optional)
cv2.imwrite('Q1/random_resized_bilinear.png', resized_img)




''' Emergency Correct code

import cv2
import numpy as np
import matplotlib.pyplot as plt

def myBilinearInterpolation(img, new_height, new_width):
    """
    Resize image using bilinear interpolation, ensuring first/last rows/columns match.
    
    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale)
    - new_height: Target number of rows (300*(M-1)+1)
    - new_width: Target number of columns (300*(N-1)+1)
    
    Returns:
    - Resized image as a NumPy array (floating-point)
    """
    height, width = img.shape[:2]
    resized = np.zeros((new_height, new_width), dtype=np.float32)
    
    # Create coordinate grids for output image
    y, x = np.indices((new_height, new_width), dtype=np.float32)
    
    # Map output coordinates to input coordinates (continuous)
    src_x = x * (width - 1) / (new_width - 1)  # 0 to 3
    src_y = y * (height - 1) / (new_height - 1)  # 0 to 3
    
    # Get integer coordinates of four neighboring pixels
    x1 = np.floor(src_x).astype(int)
    x2 = np.ceil(src_x).astype(int)
    y1 = np.floor(src_y).astype(int)
    y2 = np.ceil(src_y).astype(int)
    
    # Clip to ensure valid indices
    x1 = np.clip(x1, 0, width - 1)
    x2 = np.clip(x2, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    y2 = np.clip(y2, 0, height - 1)
    
    # Compute fractional distances
    dx = src_x - x1
    dy = src_y - y1
    
    # Handle boundaries (where x1=x2 or y1=y2)
    dx[x2 == x1] = 0
    dy[y2 == y1] = 0
    
    # Get pixel values
    f11 = img[y1, x1]
    f12 = img[y2, x1]
    f21 = img[y1, x2]
    f22 = img[y2, x2]
    
    # Compute bilinear interpolation
    resized = (f11 * (1 - dx) * (1 - dy) +
               f12 * (1 - dx) * dy +
               f21 * dx * (1 - dy) +
               f22 * dx * dy)
    
    return resized

# Load image (assuming 4x4 grayscale for testing)
image_path = 'data/interp/random.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load image at {image_path}. Using synthetic 4x4 image.")
    img = np.array([[10, 20, 30, 40],
                    [50, 60, 70, 80],
                    [90, 100, 110, 120],
                    [130, 140, 150, 160]], dtype=np.float32)
else:
    img = img.astype(np.float32)

# Verify dimensions
height, width = img.shape[:2]
if height != 4 or width != 4:
    print(f"Warning: Image is {width}x{height}, expected 4x4. Using actual dimensions.")
else:
    print("Image is 4x4 grayscale as expected.")

# Compute new dimensions
new_height = 300 * (height - 1) + 1  # 300*(4-1)+1 = 901
new_width = 300 * (width - 1) + 1   # 300*(4-1)+1 = 901
print(f"Original Dimensions: {width}x{height}")
print(f"Resized Dimensions: {new_width}x{new_height}")

# Resize using bilinear interpolation
resized_img = myBilinearInterpolation(img, new_height, new_width)

# Set colormap (jet, 256 colors)
cmap = 'jet'
norm = plt.Normalize(vmin=0, vmax=255)

# Create figure
plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
im = plt.imshow(img, cmap=cmap, norm=norm, extent=(0, width, height, 0))
plt.title(f'Original\n{width}x{height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.xticks(np.arange(0, width, 1))  # Integer ticks: 0, 1, 2, 3
plt.yticks(np.arange(0, height, 1))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

# Resized image
plt.subplot(1, 2, 2)
im = plt.imshow(resized_img, cmap=cmap, norm=norm, extent=(0, new_width, new_height, 0))
plt.title(f'Resized (Bilinear)\n{new_width}x{new_height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.xticks(np.arange(0, new_width, 100))  # Integer ticks every 100
plt.yticks(np.arange(0, new_height, 100))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.axis('on')
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

'''