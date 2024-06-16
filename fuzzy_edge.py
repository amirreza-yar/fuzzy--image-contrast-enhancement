import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PATH = Path('/home/amirreza/proj/fuzzy/image-contrast-enhancement/images/')
data = [str(p) for p in PATH.glob('*')]

# Step 1: Load the CT scan image
image_path = data[0]
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to load image '{image_path}'")
    exit()

# Step 2: Display the original image (optional)
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title('Original CT Scan Image')
plt.axis('off')
plt.show()

# Step 3: Noise Reduction - Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Example: kernel size (5, 5)

# Step 4: Display the blurred image (optional)
plt.figure(figsize=(8, 8))
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred CT Scan Image (Noise Reduction)')
plt.axis('off')
plt.show()

# Step 5: Normalization - Z-score Normalization
mean = np.mean(blurred_image)
std = np.std(blurred_image)
normalized_image = (blurred_image - mean) / std

# Step 6: Display the normalized image (optional)
plt.figure(figsize=(8, 8))
plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized CT Scan Image')
plt.axis('off')
plt.show()

# Step 7: Further Processing - Apply your fuzzy contrast enhancement or other methods
# Example:
# enhanced_image = your_fuzzy_contrast_enhancement_function(normalized_image)
# Continue with segmentation, feature extraction, etc.
