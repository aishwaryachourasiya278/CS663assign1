import cv2
import numpy as np
import matplotlib.pyplot as plt

def myCLAHE(img, num_bins=256, window_size=32, hist_threshold=0.01):
    """
    Perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the luminance channel of an RGB image.
    """
    # Step 1: Convert to YCrCb and extract luminance
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    h, w = Y.shape
    
    # Output array
    Y_clahe = np.zeros_like(Y, dtype=np.uint8)
    half_ws = window_size // 2

    for i in range(h):
        for j in range(w):
            # Step 2: Define local window
            x1, x2 = max(0, i - half_ws), min(h, i + half_ws + 1)
            y1, y2 = max(0, j - half_ws), min(w, j + half_ws + 1)
            window = Y[x1:x2, y1:y2].flatten()
            
            # Step 3: Compute histogram
            hist, _ = np.histogram(window, bins=num_bins, range=(0, 256))
            
            # Step 4: Clip histogram
            clip_limit = hist_threshold * window.size
            excess = np.sum(np.maximum(hist - clip_limit, 0))
            hist = np.minimum(hist, clip_limit)
            hist += excess // num_bins  # redistribute excess
            
            # Step 5: Compute CDF
            cdf = np.cumsum(hist).astype(np.float32)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255.0
            
            # Map intensity
            Y_clahe[i, j] = cdf[Y[i, j]]
    
    # Step 6: Recombine channels
    final = cv2.merge([Y_clahe, Cr, Cb])
    final_rgb = cv2.cvtColor(final, cv2.COLOR_YCrCb2RGB)
    return final_rgb, Y, Y_clahe

# ---- Utility to visualize ----
def show_results(original, enhanced, origY, claheY, title):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].imshow(original)
    ax[0,0].set_title("Original Image")
    ax[0,1].imshow(enhanced)
    ax[0,1].set_title(title)
    
    ax[1,0].hist(origY.ravel(), bins=256, range=(0,256), color='gray')
    ax[1,0].set_title("Original Histogram")
    ax[1,1].hist(claheY.ravel(), bins=256, range=(0,256), color='gray')
    ax[1,1].set_title("CLAHE Histogram")
    plt.show()

# Example usage:
img1 = cv2.imread("data/hist/canyon.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
enhanced1, Y1, Yc1 = myCLAHE(img1, num_bins=256, window_size=32, hist_threshold=0.01)
show_results(img1, enhanced1, Y1, Yc1, "CLAHE canyon.png")
img2 = cv2.imread("data/hist/retina.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
enhanced2, Y2, Yc2 = myCLAHE(img2, num_bins=256, window_size=32, hist_threshold=0.01)
show_results(img2 , enhanced2, Y2, Yc2, "CLAHE retina.png")
     
