import imageio.v3 as iio
import numpy as np

img = iio.imread("data/hist/leh.png").astype(np.float32)
print("dtype:", img.dtype)          # e.g. uint8, uint16, float32
print("bits per channel:", img.dtype.itemsize * 8)
print("shape:", img.shape)          # (H, W, C)
