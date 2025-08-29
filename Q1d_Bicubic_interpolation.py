import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# Load image (grayscale)
img = iio.imread('data/interp/random.png').astype(np.float32)
M, N = img.shape
M_new = 300 * (M - 1) + 1
N_new = 300 * (N - 1) + 1

# Display the image with 'jet' colormap
plt.imshow(img, extent=[0, M, 0, N], cmap='jet')
plt.colorbar(label='Intensity')
plt.title('Input Image')
plt.show()

def myBicubicInterpolation(img,M,N,M_new,N_new):
    out = np.zeros((M_new, N_new), dtype=np.float64)

    def compute_interpolation_matrix():
        rows = []
        for (X, Y) in [(0,0),(1,0),(0,1),(1,1)]:
            p, px, py, pxy = [], [], [], []
            for i in range(4):
                for j in range(4):
                    p.append(X**i * Y**j)
                    px.append(i * X**(i-1) * Y**j if i > 0 else 0)
                    py.append(j * X**i * Y**(j-1) if j > 0 else 0)
                    pxy.append(i * X**(i-1) * j * Y**(j-1) if (i>0 and j>0) else 0)
            rows.extend([p,px,py,pxy])
        return np.array(rows)

    P = compute_interpolation_matrix()
    P_inv = np.linalg.inv(P)

    for i in range(M-1):
        for j in range(N-1):
            F = []
            for (dx, dy) in [(0,0),(1,0),(0,1),(1,1)]:
                x, y = i+dx, j+dy
                F.append(img[x,y])
                # handle boundary with np.clip: np.clip(value, min, max)
                fx = 0.5*(img[np.clip(x+1,0,M-1), y] - img[np.clip(x-1,0,M-1), y])
                fy = 0.5*(img[x, np.clip(y+1,0,N-1)] - img[x, np.clip(y-1,0,N-1)])
                fxy = 0.25*(img[np.clip(x+1,0,M-1), np.clip(y+1,0,N-1)]
                          - img[np.clip(x+1,0,M-1), np.clip(y-1,0,N-1)]
                          - img[np.clip(x-1,0,M-1), np.clip(y+1,0,N-1)]
                          + img[np.clip(x-1,0,M-1), np.clip(y-1,0,N-1)])
                F.extend([fx, fy, fxy])
            F = np.array(F).reshape(-1,1)

            coeffs = P_inv @ F
            Scale_X = (M_new-1)//(M-1)
            Scale_Y = (N_new-1)//(N-1)
            for di in range(Scale_X+1 if i==M-2 else Scale_X):
                for dj in range(Scale_Y+1 if j==M-2 else Scale_Y):
                    x_local = di/Scale_X
                    y_local = dj/Scale_Y
                    val = 0.0
                    idx = 0
                    for m in range(4):
                        for n in range(4):
                            val += coeffs[idx,0] * (x_local**m) * (y_local**n)
                            idx += 1
                    out[i*Scale_X+di, j*Scale_Y+dj] = val
    return out

enlarged_img = myBicubicInterpolation(img,M,N,M_new,N_new)
print(f'Enlarged Image shape: {enlarged_img.shape}, dtype: {enlarged_img.dtype}')
plt.imshow(enlarged_img, extent=[0, M_new, 0, N_new], cmap='jet')
plt.colorbar(label='Intensity')
plt.title('Output of Bicubic interpolation')
plt.show()
