import sys
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler

def readImage(path):
    return cv2.imread(path)

def applyGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def applyBlur(image):
    return cv2.GaussianBlur(image, (21, 21), 0)

def applyMedianFilter(image, ksize):
    return cv2.medianBlur(image, ksize)

def applyGaussianFilter(image, ksize, sigma):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def prepareImageForClustering(image):
    # Reshape image to 2D array of pixels
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        pixels = image.reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 1)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_pixels = scaler.fit_transform(pixels)
    
    return scaled_pixels, (height, width)

def applyKMeans(image, n_clusters):
    pixels, dims = prepareImageForClustering(image)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape back to image dimensions
    segmented = labels.reshape(dims)
    # Normalize to 0-255 range
    segmented = ((segmented - segmented.min()) * (255 / (segmented.max() - segmented.min()))).astype(np.uint8)
    return cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

def applyKNN(image, n_neighbors):
    pixels, dims = prepareImageForClustering(image)
    # Create synthetic labels for training (using simple grid)
    y = np.random.randint(0, n_neighbors, size=len(pixels))
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(pixels, y)
    labels = knn.predict(pixels)
    
    segmented = labels.reshape(dims)
    segmented = ((segmented - segmented.min()) * (255 / (segmented.max() - segmented.min()))).astype(np.uint8)
    return cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

def applyFCM(image, n_clusters):
    pixels, dims = prepareImageForClustering(image)
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(pixels)
    labels = fcm.predict(pixels)
    
    segmented = labels.reshape(dims)
    segmented = ((segmented - segmented.min()) * (255 / (segmented.max() - segmented.min()))).astype(np.uint8)
    return cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

def createComparison(images, labels_top, labels_bottom=None):
    height, width = images[0].shape[:2]
    cols = 4  # 4x4 grid
    rows = 4
    comparison = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        y, x = row * height, col * width
        
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
            
        comparison[y:y+height, x:x+width] = image_bgr
        
        # Add top label
        if idx < len(labels_top):
            cv2.rectangle(comparison, (x + 5, y + 5), (x + 280, y + 35), (0, 0, 0), -1)
            cv2.putText(comparison, labels_top[idx], (x + 10, y + 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 2)

        # Add bottom label if provided
        if labels_bottom and idx < len(labels_bottom):
            cv2.rectangle(comparison, (x + 5, y + height - 35), (x + 280, y + height - 5), (0, 0, 0), -1)
            cv2.putText(comparison, labels_bottom[idx], (x + 10, y + height - 15),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)

    return comparison

def processImage(imagePath, outputDir):
    original = readImage(imagePath)
    
    # Preprocessing
    grayscale = applyGrayscale(original)
    median5 = applyMedianFilter(original, 5)
    gaussian = applyGaussianFilter(original, 11, 2)
    
    # Create preprocessing comparison
    preprocess_images = [original, grayscale, median5, gaussian]
    preprocess_labels = ["Original", "Grayscale", "Median (5x5)", "Gaussian"]
    preprocess_comparison = createComparison(preprocess_images, preprocess_labels)
    
    # Save preprocessing comparison
    cv2.imwrite(os.path.join(outputDir, "preprocess_comparison.jpg"), preprocess_comparison)
    
    # Apply clustering to each preprocessed image
    cluster_numbers = [2, 4, 6, 8]
    
    for image, prep_label in zip(preprocess_images, preprocess_labels):
        clustered_images = []
        cluster_labels = []
        
        for n_clusters in cluster_numbers:
            # KMeans
            kmeans_result = applyKMeans(image, n_clusters)
            clustered_images.append(kmeans_result)
            cluster_labels.append(f"KMeans (k={n_clusters})")
            
            # KNN
            knn_result = applyKNN(image, n_clusters)
            clustered_images.append(knn_result)
            cluster_labels.append(f"KNN (k={n_clusters})")
            
            # FCM
            fcm_result = applyFCM(image, n_clusters)
            clustered_images.append(fcm_result)
            cluster_labels.append(f"FCM (k={n_clusters})")
        
        # Create and save clustering comparison
        cluster_comparison = createComparison(
            clustered_images[:16],  # Take first 16 to fit 4x4 grid
            cluster_labels[:16]
        )
        
        output_filename = f"cluster_comparison_{prep_label.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(os.path.join(outputDir, output_filename), cluster_comparison)

if __name__ == "__main__":
    inputImagePath = sys.argv[1]
    outputDirectory = sys.argv[2]
    processImage(inputImagePath, outputDirectory)