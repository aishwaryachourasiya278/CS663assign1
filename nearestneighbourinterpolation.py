# Starter code to load libraries, read an image, and display with 'jet' colormap

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# Load an example image (replace 'your_image.tif' with your file)
img = iio.imread('data/interp/random.png').astype(np.float32)
print(f'Image shape: {img.shape}, dtype: {img.dtype}')
M = img.shape[0]
N = img.shape[1]
M_new = 300*(M-1)+1
N_new = 300*(N-1)+1

# Display the image with 'jet' colormap
plt.imshow(img, extent=[0, N, 0, M], origin='upper', cmap='jet')
plt.colorbar(label='Intensity')
plt.title('Image with Jet Colormap')
plt.show()

def myNearestNeighborInterpolation(img):
    Scale_X = (M-1)/(M_new-1)
    Scale_Y = (N-1)/(N_new-1)
    enlarged_img = np.zeros((M_new,N_new), dtype=img.dtype)
    for i in range(M_new):
        for j in range(N_new):
            x = round(i*Scale_X)
            y = round(j*Scale_Y)
            enlarged_img[i,j] = img[x,y]
    return enlarged_img

enlarged_img = myNearestNeighborInterpolation(img)
print(f'Enlarged Image shape: {enlarged_img.shape}, dtype: {enlarged_img.dtype}')


# Display the image with 'jet' colormap
plt.imshow(enlarged_img, extent=[0, N_new, 0, M_new], origin='upper', cmap='jet')
plt.colorbar(label='Intensity')
plt.title('Image with Jet Colormap')
plt.show()

    




