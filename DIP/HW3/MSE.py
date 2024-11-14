import sys
import cv2
import numpy as np

def loadImage(imagePath):
    return cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

def calcMSE(image1, image2):
    #super easy lol
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

def mseFromPaths(imagePath1, imagePath2):
    return calcMSE(loadImage(imagePath1), loadImage(imagePath2))

def main():
    imagePath1, imagePath2 = sys.argv[1], sys.argv[2]
    print(f"MSE: {mseFromPaths(imagePath1, imagePath2)}")

if __name__ == "__main__":
    main()
