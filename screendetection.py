import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def display_image(image, title="Image"):
    """Display an image using Matplotlib."""
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.title(title)
    plt.show()

def skin_detection_kmeans(image_path, k_clusters=2):
    # Load the image and convert from BGR (OpenCV format) to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for faster processing
    resized_image = cv2.resize(image, (300, 300))
    
    # Convert from RGB to HSV for better color separation
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
    
    # Reshape for clustering
    pixels = hsv_image.reshape((-1, 3))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Extract labels and centroids
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Reshape labels back to image dimensions
    segmented_image = labels.reshape(resized_image.shape[:2])
    
    # Analyze cluster centroids to find the skin cluster
    skin_cluster_index = np.argmin(np.linalg.norm(centers - np.array([160, 100, 80]), axis=1))  # Approximate skin HSV
    
    # Create binary mask
    mask = (segmented_image == skin_cluster_index).astype(np.uint8) * 255
    
    # Refine the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask to the original image
    skin_detected_image = cv2.bitwise_and(resized_image, resized_image, mask=refined_mask)
    
    # Display results
    display_image(resized_image, "Original Image")
    display_image(refined_mask, "Refined Skin Mask (Binary)")
    display_image(skin_detected_image, "Detected Skin Regions")

# Test with your image
skin_detection_kmeans("shakib2.jpg")
