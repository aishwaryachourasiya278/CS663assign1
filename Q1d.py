import cv2
import numpy as np
import matplotlib.pyplot as plt


def myBicubicInterpolation(img, new_height, new_width):
    """
    Resize image using bicubic interpolation, ensuring first/last rows/columns match.

    Parameters:
    - img: Input image (NumPy array, floating-point, grayscale)
    - new_height: Target number of rows (300*(M-1)+1)
    - new_width: Target number of columns (300*(N-1)+1)

    Returns:
    - Resized image as a NumPy array (floating-point)
    """
    height, width = img.shape
    num_cells_y = height - 1
    num_cells_x = width - 1
    scale = (new_height - 1) // num_cells_y  # Assuming integer scale, here 300

    # Step 1: Compute derivatives
    def compute_derivatives(img):
        h, w = img.shape
        fx = np.zeros_like(img)
        fy = np.zeros_like(img)
        fxy = np.zeros_like(img)

        # fx
        for y in range(h):
            for x in range(w):
                if x == 0:
                    fx[y, x] = img[y, x+1] - img[y, x]
                elif x == w-1:
                    fx[y, x] = img[y, x] - img[y, x-1]
                else:
                    fx[y, x] = 0.5 * (img[y, x+1] - img[y, x-1])

        # fy
        for x in range(w):
            for y in range(h):
                if y == 0:
                    fy[y, x] = img[y+1, x] - img[y, x]
                elif y == h-1:
                    fy[y, x] = img[y, x] - img[y-1, x]
                else:
                    fy[y, x] = 0.5 * (img[y+1, x] - img[y-1, x])

        # fxy
        for y in range(h):
            for x in range(w):
                xp1 = min(x+1, w-1)
                xm1 = max(x-1, 0)
                yp1 = min(y+1, h-1)
                ym1 = max(y-1, 0)
                fxy[y, x] = 0.25 * ((img[yp1, xp1] + img[ym1, xm1]) -
                                    (img[yp1, xm1] + img[ym1, xp1]))

        return fx, fy, fxy

    fx, fy, fxy = compute_derivatives(img)

    # Step 2: Fixed A matrix 16x16
    A = np.zeros((16, 16))

    def set_equations(row_start, s, t):
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                A[row_start + 0, col] = s**i * t**j
                if i >= 1:
                    A[row_start + 1, col] = i * s**(i-1) * t**j
                if j >= 1:
                    A[row_start + 2, col] = j * s**i * t**(j-1)
                if i >= 1 and j >= 1:
                    A[row_start + 3, col] = i * j * s**(i-1) * t**(j-1)

    set_equations(0, 0, 0)
    set_equations(4, 1, 0)
    set_equations(8, 0, 1)
    set_equations(12, 1, 1)

    # Step 3: Compute coefficients for each cell
    coefficients = np.zeros((num_cells_y, num_cells_x, 16))

    for ky in range(num_cells_y):
        for kx in range(num_cells_x):
            b = np.array([
                img[ky, kx], fx[ky, kx], fy[ky, kx], fxy[ky, kx],
                img[ky, kx+1], fx[ky, kx+1], fy[ky, kx+1], fxy[ky, kx+1],
                img[ky+1, kx], fx[ky+1, kx], fy[ky+1, kx], fxy[ky+1, kx],
                img[ky+1, kx+1], fx[ky+1, kx+1], fy[ky+1, kx+1], fxy[ky+1, kx+1]
            ])
            a = np.linalg.solve(A, b)
            coefficients[ky, kx] = a

    # Step 4: Create resized image
    resized = np.zeros((new_height, new_width))

    for ky in range(num_cells_y):
        start_row = ky * scale
        end_row = (ky + 1) * scale + (1 if ky == num_cells_y - 1 else 0)
        for kx in range(num_cells_x):
            start_col = kx * scale
            end_col = (kx + 1) * scale + (1 if kx == num_cells_x - 1 else 0)

            # Local mesh
            r_local, c_local = np.meshgrid(
                np.arange(start_row, end_row),
                np.arange(start_col, end_col),
                indexing='ij'
            )

            # Fractional
            frac_x = (c_local / float(scale)) - kx
            frac_y = (r_local / float(scale)) - ky

            # Powers
            powers_x = np.stack([frac_x**0, frac_x**1, frac_x**2, frac_x**3], axis=-1)
            powers_y = np.stack([frac_y**0, frac_y**1, frac_y**2, frac_y**3], axis=-1)

            # Coefficients reshaped to (4,4)
            a_resh = coefficients[ky, kx].reshape(4, 4)

            # Einsum
            val = np.einsum('ij,abi,abj->ab', a_resh, powers_x, powers_y)

            # Assign
            resized[start_row:end_row, start_col:end_col] = val

    return np.clip(resized, 0, 255)


# ---------- Run test ----------
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
new_width = 300 * (width - 1) + 1    # 300*(4-1)+1 = 901
print(f"Original Dimensions: {width}x{height}")
print(f"Resized Dimensions: {new_width}x{new_height}")

# Resize using bicubic interpolation
resized_img = myBicubicInterpolation(img, new_height, new_width)

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
plt.xticks(np.arange(0, width, 1))
plt.yticks(np.arange(0, height, 1))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.gca().set_aspect('equal', adjustable='box')

# Resized image
plt.subplot(1, 2, 2)
im = plt.imshow(resized_img, cmap=cmap, norm=norm, extent=(0, new_width, new_height, 0))
plt.title(f'Resized\n{new_width}x{new_height}')
plt.xlabel('Pixels (X)')
plt.ylabel('Pixels (Y)')
plt.xticks(np.arange(0, new_width, 100))
plt.yticks(np.arange(0, new_height, 100))
plt.colorbar(im, label='Intensity', fraction=0.046, pad=0.04)
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Save resized image (optional)
cv2.imwrite('Q1/random_resized_bicubic.png', resized_img)
