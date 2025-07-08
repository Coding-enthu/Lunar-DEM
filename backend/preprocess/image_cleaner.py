import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, blur_ksize=3):
    # Step 1: Load in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image at {image_path}")
    
    # Step 2: Resize for faster processing (optional)
    img_gray = cv2.resize(img_gray, (256, 256))
    
    # Step 3: Normalize intensity to [0, 1]
    img_norm = img_gray.astype(np.float32) / 255.0
    
    # Step 4: Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_norm, (blur_ksize, blur_ksize), 0)
    
    return img_blur

# Optional: Show the image
if __name__ == "__main__":
    processed = preprocess_image("images/lunar_sample.png")
    print("Image min/max after preprocessing:", processed.min(), processed.max())
    plt.imshow(processed, cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()
