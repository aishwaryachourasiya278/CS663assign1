import cv2
import numpy as np
import matplotlib.pyplot as plt

def myHistMatch(src, ref, num_bins=256):
    """
    Perform histogram matching of src image to match ref image
    on luminance + chroma (ignoring black background).
    """
    # Convert to YCrCb
    src_ycc = cv2.cvtColor(src, cv2.COLOR_RGB2YCrCb)
    ref_ycc = cv2.cvtColor(ref, cv2.COLOR_RGB2YCrCb)

    src_Y, src_Cr, src_Cb = cv2.split(src_ycc)
    ref_Y, ref_Cr, ref_Cb = cv2.split(ref_ycc)

    # Create foreground masks (ignore black pixels)
    src_mask = np.any(src > 0, axis=-1)
    ref_mask = np.any(ref > 0, axis=-1)

    # Function to match histograms for one channel
    def match_channel(src_chan, ref_chan, src_mask, ref_mask, num_bins):
        # Flatten masked pixels
        src_vals = src_chan[src_mask].ravel()
        ref_vals = ref_chan[ref_mask].ravel()

        # Compute histograms
        src_hist, bins = np.histogram(src_vals, bins=num_bins, range=(0,256), density=True)
        ref_hist, _    = np.histogram(ref_vals, bins=num_bins, range=(0,256), density=True)

        # Compute CDFs
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        # Build mapping: for each src intensity, find closest ref intensity
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            diff = np.abs(src_cdf[i * (num_bins//256) : i * (num_bins//256) + 1] - ref_cdf)
            mapping[i] = np.argmin(diff)

        # Apply mapping only to foreground
        matched = src_chan.copy()
        matched[src_mask] = mapping[src_chan[src_mask]]
        return matched

    # Match each channel
    Y_matched  = match_channel(src_Y, ref_Y, src_mask, ref_mask, num_bins)
    Cr_matched = match_channel(src_Cr, ref_Cr, src_mask, ref_mask, num_bins)
    Cb_matched = match_channel(src_Cb, ref_Cb, src_mask, ref_mask, num_bins)

    # Merge channels
    matched_ycc = cv2.merge([Y_matched, Cr_matched, Cb_matched])
    matched_rgb = cv2.cvtColor(matched_ycc, cv2.COLOR_YCrCb2RGB)

    return matched_rgb, (src_Y, Y_matched, ref_Y)


# --------- Visualization ----------
def show_hist_match(src, ref, matched, srcY, matchedY, refY):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # Images
    ax[0,0].imshow(src); ax[0,0].set_title("Source")
    ax[0,1].imshow(ref); ax[0,1].set_title("Reference")
    ax[0,2].imshow(matched); ax[0,2].set_title("Histogram Matched")

    # Histograms (Y channel)
    ax[1,0].hist(srcY.ravel(), bins=256, range=(0,256), color="blue")
    ax[1,0].set_title("Source Y Histogram")
    ax[1,1].hist(refY.ravel(), bins=256, range=(0,256), color="green")
    ax[1,1].set_title("Reference Y Histogram")
    ax[1,2].hist(matchedY.ravel(), bins=256, range=(0,256), color="red")
    ax[1,2].set_title("Matched Y Histogram")

    plt.tight_layout()
    plt.show()


src = cv2.imread("data/hist/retina.png")
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
ref = cv2.imread("data/hist/retinaRef.png")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
matched, (srcY, matchedY, refY) = myHistMatch(src, ref, num_bins=128)
show_hist_match(src, ref, matched, srcY, matchedY, refY)
