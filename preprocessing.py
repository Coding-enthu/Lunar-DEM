import cv2
import numpy as np

def preprocess_image(path, size=(256, 256)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32)/255.0
    return img