import cv2
import numpy as np
import sys
from MSE import mseFromPaths, loadImage, calcMSE

def applyGaussianFilter(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def applyMedianFilter(image, kernelSize):
    return cv2.medianBlur(image, kernelSize)

def calculateFilteredMSE(originalImage, noisyImage, filterFunction, *args):
    filteredImage = filterFunction(noisyImage, *args)
    return calcMSE(originalImage, filteredImage), filteredImage

def generateComparisonTable(originalPath, noisyPath):
    originalImage = loadImage(originalPath)
    noisyImage = loadImage(noisyPath)

    mseNoisy = mseFromPaths(originalPath, noisyPath)
    mseGaussianSigma2, gaussianSigma2Image = calculateFilteredMSE(originalImage, noisyImage, applyGaussianFilter, 2)
    mseGaussianSigma7, gaussianSigma7Image = calculateFilteredMSE(originalImage, noisyImage, applyGaussianFilter, 7)
    mseMedian7x7, median7x7Image = calculateFilteredMSE(originalImage, noisyImage, applyMedianFilter, 7)
    mseMedian19x19, median19x19Image = calculateFilteredMSE(originalImage, noisyImage, applyMedianFilter, 19)

    return {
        "Noisy (not processed)": (mseNoisy, noisyImage),
        "Gaussian Filter (sigma=2)": (mseGaussianSigma2, gaussianSigma2Image),
        "Gaussian Filter (sigma=7)": (mseGaussianSigma7, gaussianSigma7Image),
        "Median Filter (7x7)": (mseMedian7x7, median7x7Image),
        "Median Filter (19x19)": (mseMedian19x19, median19x19Image)
    }

def printComparisonTable(comparisonResults):
    print("MSE Comparison Table")
    print("-------------------")
    for description, (mseValue, _) in comparisonResults.items():
        print(f"{description}: {mseValue:.2f}")

def createImageGrid(images):
    rows = []
    for i in range(0, len(images), 2):
        row = np.hstack(images[i:i+2])
        rows.append(row)
    return np.vstack(rows)

def saveFilteredImages(comparisonResults, outputPath):
    images = [result[1] for result in list(comparisonResults.values())[1:]]  # Exclude the noisy image
    titles = list(comparisonResults.keys())[1:]  # Exclude the noisy image title
    mse_values = [result[0] for result in list(comparisonResults.values())[1:]]  # Exclude the noisy image MSE
    
    gridImage = createImageGrid(images)
    
    # Add titles and MSE values to the grid image
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for i, (title, mse) in enumerate(zip(titles, mse_values)):
        x = 10 + (i % 2) * (gridImage.shape[1] // 2)
        y = y_offset + (i // 2) * (gridImage.shape[0] // 2)
        # Draw a black background rectangle for the text
        label = f"{title} MSE: {mse:.2f}"
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.rectangle(gridImage, (x, y - text_size[1] - 5), (x + text_size[0], y + 5), (0, 0, 0), -1)
        # Draw the white text on top of the black background
        cv2.putText(gridImage, label, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imwrite(outputPath, gridImage)

def main():
    originalPath, noisyPath = sys.argv[1], sys.argv[2]
    comparisonResults = generateComparisonTable(originalPath, noisyPath)
    printComparisonTable(comparisonResults)
    saveFilteredImages(comparisonResults, "filtered_images_grid.png")

if __name__ == "__main__":
    main()
