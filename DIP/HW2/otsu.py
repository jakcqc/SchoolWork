import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def computeOtsuThresholdRecursive(
    hist,
    total_pixels,
    total_mean_intensity,
    threshold=0,
    background_weight=0,
    background_sum=0,
    max_between_class_variance=0,
    optimal_threshold=0
):
   
    if threshold > 255:
        return optimal_threshold

    # ++ background and current b sum 
    background_weight += hist[threshold]
    background_sum += threshold * hist[threshold]
    #base case
    if background_weight == 0:
        return computeOtsuThresholdRecursive(
            hist, total_pixels, total_mean_intensity,
            threshold + 1, background_weight, background_sum,
            max_between_class_variance, optimal_threshold
        )

    # Compute foreground
    foreground_weight = total_pixels - background_weight

    # recursive end decision 
    if foreground_weight == 0:
        return optimal_threshold

    # means lol
    mean_background = background_sum / background_weight
    mean_foreground = (total_mean_intensity - background_sum) / foreground_weight

    # between-class variance 
    between_class_variance = (
        background_weight
        * foreground_weight
        * (mean_background - mean_foreground) ** 2
    )

    # Update the optimal threshold if a new maximum is found
    if between_class_variance > max_between_class_variance:
        max_between_class_variance = between_class_variance
        optimal_threshold = threshold

    # Recursively evaluate the next threshold
    return computeOtsuThresholdRecursive(
        hist, total_pixels, total_mean_intensity,
        threshold + 1, background_weight, background_sum,
        max_between_class_variance, optimal_threshold
    )

def computeOtsuThresholdIterative(hist, total_pixels):
    
    # same as recursive just wrapped in a loop
    total_mean_intensity = np.dot(np.arange(256), hist)

    background_weight = 0   
    background_sum = 0     
    max_between_class_variance = 0
    optimal_threshold = 0

    for threshold in range(256):
        background_weight += hist[threshold]
        background_sum += threshold * hist[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total_pixels - background_weight
        if foreground_weight == 0:
            break
        mean_background = background_sum / background_weight
        mean_foreground = (total_mean_intensity - background_sum) / foreground_weight
        between_class_variance = (
            background_weight
            * foreground_weight
            * (mean_background - mean_foreground) ** 2
        )
        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            optimal_threshold = threshold
    return optimal_threshold

def otsuThresholdingRecursive(image,channel = 0):
   
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]).flatten().astype(np.int32)
    total_pixels = image.size

    # dot on our hist values and even space n arranged values
    total_mean_intensity = np.dot(np.arange(256), hist)
    threshold = computeOtsuThresholdRecursive(hist, total_pixels, total_mean_intensity)

    # binary based on assignment 0 or 1 for background forground
    binary_image = (image > threshold).astype(np.uint8)
    return threshold, binary_image, hist

def otsuThresholdingIterative(image, channel = 0):
   
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]).flatten().astype(np.int32)
    total_pixels = image.size
    threshold = computeOtsuThresholdIterative(hist, total_pixels)
    binary_image = (image > threshold).astype(np.uint8)
    return threshold, binary_image, hist

def plotHistogram(hist):
    plt.figure(figsize=(10, 5))
    plt.title("Intensity Histogram")
    plt.xlabel("Pix Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black')
    plt.xlim([0, 255])
    plt.grid(True)
    plt.show()

def main():
    #for image path
    imagePath = sys.argv[1]
    image = cv2.imread(imagePath)
    plt.figure(figsize=(6, 6))
    if image.ndim == 3 and image.shape[2] == 3:
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(imageRGB)
    else:
        plt.imshow(grayImage, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    for ch in range(3):
        if image.ndim == 3 and image.shape[2] == 3:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayImage = image[:,:,ch]
        else:
            grayImage = image.copy()
        

        startRecursive = time.time()
        thresholdRecursive, binaryImageRecursive, histRecursive = otsuThresholdingRecursive(grayImage)
        endRecursive = time.time()
        timeRecursive = endRecursive - startRecursive

        startIterative = time.time()
        thresholdIterative, binaryImageIterative, histIterative = otsuThresholdingIterative(grayImage)
        endIterative = time.time()
        timeIterative = endIterative - startIterative

        


        plotHistogram(histRecursive)

        print(f"Recursive Threshold: {thresholdRecursive}, Time Taken: {timeRecursive:.6f} seconds")
        print(f"Iterative Threshold: {thresholdIterative}, Time Taken: {timeIterative:.6f} seconds")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(binaryImageRecursive, cmap='gray')
        plt.title(f"Recursive Binary Image\nThreshold = {thresholdRecursive}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(binaryImageIterative, cmap='gray')
        plt.title(f"Iterative Binary Image\nThreshold = {thresholdIterative}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
