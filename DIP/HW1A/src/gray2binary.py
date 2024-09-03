
import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import os


if len(sys.argv) != 4:
    print("Usage: gray2binary.py <inputfilename> <outputfilename> <thresholdvalue>")
    sys.exit(1)

# get in and out args
input_name = sys.argv[1]
output_name = sys.argv[2]
threshold = float(sys.argv[3])
print("Args: ", input_name,", ",output_name, "," , threshold)

#read image in using cv and should be gray
img = cv.imread(input_name, cv.IMREAD_GRAYSCALE)
# use beast mode np.where which goes over the whole container and does an if + return 255 or 0
img = np.where(img > threshold,255,0)
cv.imwrite(output_name,img)