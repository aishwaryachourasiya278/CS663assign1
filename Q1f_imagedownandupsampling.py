import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# load .mat file
data = scipy.io.loadmat("data/interp/ct.mat")
original = data["original"]
subsampled = data["subsampled"]

M, N = subsampled.shape
M_new, N_new = original.shape

# nearest neighbor interpolation
def myNearestNeighborInterpolation(img):
    Scale_X = (M-1)/(M_new-1)
    Scale_Y = (N-1)/(N_new-1)
    enlarged_img = np.zeros((M_new, N_new), dtype=img.dtype)
    for i in range(M_new):
        for j in range(N_new):
            x = round(i*Scale_X)
            y = round(j*Scale_Y)
            enlarged_img[i, j] = img[x, y]
    return enlarged_img

# bilinear interpolation
def myBilinearInterpolation(img, M, N, M_new, N_new):
    Scale_X = (M-1)/(M_new-1)
    Scale_Y = (N-1)/(N_new-1)
    out = np.zeros((M_new, N_new), dtype=img.dtype)
    for i in range(M_new):
        for j in range(N_new):
            x = i * Scale_X
            y = j * Scale_Y
            x0, x1 = int(np.floor(x)), int(np.ceil(x))
            y0, y1 = int(np.floor(y)), int(np.ceil(y))
            dx, dy = x - x0, y - y0
            # clamp to avoid index errors at boundary
            x1 = min(x1, M-1)
            y1 = min(y1, N-1)
            # bilinear formula
            val = (1-dx)*(1-dy)*img[x0,y0] + dx*(1-dy)*img[x1,y0] \
                  + (1-dx)*dy*img[x0,y1] + dx*dy*img[x1,y1]
            out[i,j] = val
    return out


# bicubic interpolation
def myBicubicInterpolation(img, M, N, M_new, N_new):
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
                for dj in range(Scale_Y+1 if j==N-2 else Scale_Y):
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

# run all interpolations
nn_img = myNearestNeighborInterpolation(subsampled)
bilinear_img = myBilinearInterpolation(subsampled, M, N, M_new, N_new)
bicubic_img = myBicubicInterpolation(subsampled, M, N, M_new, N_new)

# compute differences
diff_nn = original - nn_img
diff_bilinear = original - bilinear_img
diff_bicubic = original - bicubic_img

# RMSE function
def rmse(gt, pred):
    return np.sqrt(np.mean((gt - pred)**2))

# compute RMSE values
rmse_nn = rmse(original, nn_img)
rmse_bilinear = rmse(original, bilinear_img)
rmse_bicubic = rmse(original, bicubic_img)

# show original + enlarged with RMSE
vmin = min(original.min(), nn_img.min(), bilinear_img.min(), bicubic_img.min())
vmax = max(original.max(), nn_img.max(), bilinear_img.max(), bicubic_img.max())

plt.figure(figsize=(14, 12))

plt.subplot(2,2,1)
im1 = plt.imshow(original, cmap='jet', vmin=vmin, vmax=vmax)
plt.title("Original")
plt.colorbar(im1, fraction=0.046, pad=0.04)

plt.subplot(2,2,2)
im2 = plt.imshow(nn_img, cmap='jet', vmin=vmin, vmax=vmax)
plt.title(f"Nearest (RMSE={rmse_nn:.2f})")
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.subplot(2,2,3)
im3 = plt.imshow(bilinear_img, cmap='jet', vmin=vmin, vmax=vmax)
plt.title(f"Bilinear (RMSE={rmse_bilinear:.2f})")
plt.colorbar(im3, fraction=0.046, pad=0.04)

plt.subplot(2,2,4)
im4 = plt.imshow(bicubic_img, cmap='jet', vmin=vmin, vmax=vmax)
plt.title(f"Bicubic (RMSE={rmse_bicubic:.2f})")
plt.colorbar(im4, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# show difference images with RMSE
dmin = min(diff_nn.min(), diff_bilinear.min(), diff_bicubic.min())
dmax = max(diff_nn.max(), diff_bilinear.max(), diff_bicubic.max())

plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
im5 = plt.imshow(diff_nn, cmap='jet', vmin=dmin, vmax=dmax)
plt.title(f"Diff Nearest (RMSE={rmse_nn:.2f})")
plt.colorbar(im5, fraction=0.046, pad=0.04)

plt.subplot(1,3,2)
im6 = plt.imshow(diff_bilinear, cmap='jet', vmin=dmin, vmax=dmax)
plt.title(f"Diff Bilinear (RMSE={rmse_bilinear:.2f})")
plt.colorbar(im6, fraction=0.046, pad=0.04)

plt.subplot(1,3,3)
im7 = plt.imshow(diff_bicubic, cmap='jet', vmin=dmin, vmax=dmax)
plt.title(f"Diff Bicubic (RMSE={rmse_bicubic:.2f})")
plt.colorbar(im7, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

