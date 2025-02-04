import numpy as np
import cv2


def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
 
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
       
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

       
        new_centroids = np.array([
            data[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i) else centroids[i]
            for i in range(k)
        ])

       
        if np.allclose(centroids, new_centroids, atol=tol):
            break

        centroids = new_centroids

    return centroids, cluster_assignments


def find_optimal_clusters(data, max_k=10):
    max_k = min(max_k, len(data))
    wcss = []  

    for k in range(2, max_k + 1):
        centroids, cluster_assignments = kmeans_clustering(data, k)
       
        wcss.append(np.sum((data - centroids[cluster_assignments]) ** 2))

   
    if len(wcss) < 3:
        return 2  

    second_derivative = np.diff(wcss, 2)
    elbow_k = np.argmin(second_derivative) + 2

    return elbow_k


def segment_image(image_path, output_path):
   
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

   
    
    optimal_k = 2
    print(f"Optimal number of clusters: {optimal_k}")

   
    centroids, cluster_assignments = kmeans_clustering(pixels, optimal_k)

    
    segmented_pixels = centroids[cluster_assignments].astype(np.uint8)
    segmented_image = segmented_pixels.reshape(image_rgb.shape)

    
    segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, segmented_bgr)

    print(f"Segmented image saved as {output_path}")


if __name__ == "__main__":
    segment_image("amla.jpeg", "2_segmented_image.jpeg")
