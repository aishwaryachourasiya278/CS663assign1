import cv2
import numpy as np
import matplotlib.pyplot as plt

def myHistEqualize(img):
    # Check if image is color
    is_color = len(img.shape) == 3 and img.shape[2] == 3
    
    # Step 1: Convert to YCrCb if color, otherwise use grayscale
    if is_color:
        # Convert RGB to YCrCb
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb_img)
    else:
        Y = img.copy()
    
    # Step 2: Compute histogram of Y
    hist, bins = np.histogram(Y.flatten(), bins=256, range=[0, 256])
    
    # Normalize histogram to probability distribution
    p = hist / np.sum(hist)
    
    # Step 3: Compute CDF
    cdf = np.cumsum(p)
    
    # Step 4: Create mapping function
    new_Y_mapping = np.round(cdf * 255).astype(np.uint8)
    
    # Step 5: Apply mapping to Y channel
    Y_eq = new_Y_mapping[Y]
    
    # Step 6: Process based on color/grayscale
    if is_color:
        # Merge Y_eq with Cr and Cb
        enhanced_ycrcb = cv2.merge([Y_eq, Cr, Cb])
        # Convert back to RGB
        enhanced_img = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        enhanced_img = Y_eq
    
    return enhanced_img, Y, Y_eq, hist

def display_results(original_img, enhanced_img, Y, Y_eq, original_hist):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Display original image
    if len(original_img.shape) == 3:
        axes[0, 0].imshow(original_img)
    else:
        axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Display enhanced image
    if len(enhanced_img.shape) == 3:
        axes[0, 1].imshow(enhanced_img)
    else:
        axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title('Enhanced Image')
    axes[0, 1].axis('off')
    
    # Display original histogram of Y
    axes[1, 0].bar(range(256), original_hist, width=1.0)
    axes[1, 0].set_title('Histogram of Original Y Channel')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim([0, 255])
    
    # Display enhanced histogram of Y_eq
    enhanced_hist, _ = np.histogram(Y_eq.flatten(), bins=256, range=[0, 256])
    axes[1, 1].bar(range(256), enhanced_hist, width=1.0)
    axes[1, 1].set_title('Histogram of Enhanced Y Channel')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim([0, 255])
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load an image (replace with your image path)
    # For color image:
    # img = cv2.imread('your_image.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # For demonstration, let's create a sample image
    # Create a low-contrast grayscale image
    gray_img = np.random.randint(50, 150, (200, 200), dtype=np.uint8)
    
    # Create a low-contrast color image
    color_img = np.zeros((200, 200, 3), dtype=np.uint8)
    color_img[:, :, 0] = np.random.randint(50, 150, (200, 200))  # R
    color_img[:, :, 1] = np.random.randint(50, 150, (200, 200))  # G
    color_img[:, :, 2] = np.random.randint(50, 150, (200, 200))  # B
    
    # Test with grayscale image
    print("Testing with grayscale image...")
    enhanced_gray, Y_gray, Y_eq_gray, hist_gray = myHistEqualize(gray_img)
    display_results(gray_img, enhanced_gray, Y_gray, Y_eq_gray, hist_gray)
    
    # Test with color image
    print("Testing with color image...")
    enhanced_color, Y_color, Y_eq_color, hist_color = myHistEqualize(color_img)
    display_results(color_img, enhanced_color, Y_color, Y_eq_color, hist_color)