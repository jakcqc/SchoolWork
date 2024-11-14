import cv2
import numpy as np
import sys
from MSE import mseFromPaths, loadImage, calcMSE

def applyGaussianFilter(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def applyMedianFilter(image, kernelSize):
    return cv2.medianBlur(image, kernelSize)
def applyAdaptiveMedianFilter(image, sMax):
    import time
    start_time = time.time()
    reconstructionAmount = 0

    def processPixel(x, y, s):
        nonlocal reconstructionAmount
        if s > sMax:
            return image[y, x]
        
        #calc the window
        #grayscale so 2d use the current x and y to compute the coords/bounds of window
        halfSize = s // 2
        top = max(y - halfSize, 0)
        bottom = min(y + halfSize + 1, image.shape[0])
        left = max(x - halfSize, 0)
        right = min(x + halfSize + 1, image.shape[1])
        
        #get splice using window calcs since we just need one x,y coord
        window = image[top:bottom, left:right]
        #values for median calc
        zMin = np.min(window)
        zMed = np.median(window)
        zMax = np.max(window)
        zXy = image[y, x]
        
        #impulse check
        #basically whether we accept or reject the change
        if zMin < zMed < zMax:
            # since we check that median value is actually representative of the window
            # then current pix is good when it is betwen min max which would be noise in this case (the two noise classses we care about)
            if zMin < zXy < zMax:
                #orignal is good
                return zXy
            else:
                #need the median now :) aka reconstruction
                reconstructionAmount += 1
                return zMed
        else:
            #recursive call for next window size
            # here we increase filter by odd number 
            # -- since we are not doing a down sample etc. and need a center pixel for median
            return processPixel(x, y, s + 2)
    
    height, width = image.shape
    #create result arr to fill up with median processed pix
    result = np.zeros((height, width), dtype=np.uint8)
    
    #go over image now
    for y in range(height):
        for x in range(width):
            #give window start size 3 here
            result[y, x] = processPixel(x, y, 3)
    
    end_time = time.time()
    print(f"Adaptive Median Filter (Smax={sMax}x{sMax}) took {end_time - start_time:.2f} seconds, with {reconstructionAmount} reconstructions")
    
    return result
#func programming passing func to call on args
def calculateFilteredMSE(originalImage, noisyImage, filterFunction, *args):
    filteredImage = filterFunction(noisyImage, *args)
    return calcMSE(originalImage, filteredImage), filteredImage

def generateComparisonTable(originalPath, noisyPath):
    originalImage = loadImage(originalPath)
    noisyImage = loadImage(noisyPath)

    return {
        "Noisy and OG MSE": (mseFromPaths(originalPath, noisyPath), noisyImage),
        "Adaptive Median (Smax=7x7) from 3x3": calculateFilteredMSE(originalImage, noisyImage, applyAdaptiveMedianFilter, 7),
        "Adaptive Median (Smax=19x19) from 3x3": calculateFilteredMSE(originalImage, noisyImage, applyAdaptiveMedianFilter, 19),
    }

def printComparisonTable(comparisonResults):
    print("MSE Table")
    for description, (mseValue, _) in comparisonResults.items():
        print(f"{description}: {mseValue:.2f}")


def createImageGrid(images):
    #ty numpy lol 
    rows = [np.hstack(images[i:i+2]) for i in range(0, len(images), 2)]
    return np.vstack(rows)

def saveFilteredImages(comparisonResults, outputPath):
    images = [result[1] for result in list(comparisonResults.values())[1:]]
    titles = list(comparisonResults.keys())[1:]
    mseValues = [result[0] for result in list(comparisonResults.values())[1:]]
    
    gridImage = createImageGrid(images)
    
    #sample code from opencv tool to put text on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    yOffset = 30
    for i, (title, mse) in enumerate(zip(titles, mseValues)):
        x = 10 + (i % 2) * (gridImage.shape[1] // 2)
        y = yOffset + (i // 2) * (gridImage.shape[0] // 2)
        label = f"{title} MSE: {mse:.2f}"
        textSize = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.rectangle(gridImage, (x, y - textSize[1] - 5), (x + textSize[0], y + 5), (0, 0, 0), -1)
        cv2.putText(gridImage, label, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imwrite(outputPath, gridImage)

def main():
    originalPath, noisyPath = sys.argv[1], sys.argv[2]
    comparisonResults = generateComparisonTable(originalPath, noisyPath)
    printComparisonTable(comparisonResults)
    saveFilteredImages(comparisonResults, "adaptive_median_comparison.png")

if __name__ == "__main__":
    main()