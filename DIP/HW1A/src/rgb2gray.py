
import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import os


if len(sys.argv) != 3:
    print("Usage: rgb2gray.py <arg1> <arg2>")
    sys.exit(1)

# get in and out args
input_name = sys.argv[1]
output_name = sys.argv[2]

print("Args: ", input_name,", ",output_name)

#read image in using cv
img = cv.imread(input_name, cv.IMREAD_COLOR)
#our ground truth reference 
imgG = cv.imread(input_name, cv.IMREAD_GRAYSCALE)
cv.imwrite("groundTruthGray_" + output_name,imgG)

def convertToGray(img,output_loc):
    print("Image details: ")
    print(type(img))     
    print(img.shape)      
    print(img.dtype) 
    
    # values from outline
    blue_scalar = 0.1140  
    green_scalar = 0.5871  
    red_scalar = 0.2989  

    # use py slice very cool to mult by the scalar using the values from homework outline
    img[:, :, 0] = np.clip(img[:, :, 0] * blue_scalar, 0, 255)  
    img[:, :, 1] = np.clip(img[:, :, 1] * green_scalar, 0, 255)  
    img[:, :, 2] = np.clip(img[:, :, 2] * red_scalar, 0, 255)  

    #again, use the slice to add all channels together 
    newGray = img[:,:,0] + img[:,:,1] + img[:,:,2]
    
    #write out using the outname
    cv.imwrite(output_loc,newGray)

#here is the code from opencv for simple operations on threshold and plotting 
def plotThresh(img, img_name="output"):
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Apply the different thresholding techniques
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    
    # Titles and corresponding images
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    
    # Create the images directory if it doesn't exist
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save each image
    for i in range(6):
        plt.figure(figsize=(6, 6))
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        
        # Save the plot to the images folder
        file_path = os.path.join(output_dir, f"{img_name}_{titles[i]}.png")
        plt.savefig(file_path)
        plt.close()

    print(f"Images saved to {output_dir}/")

#call our funcs
# Gray[x][y]= 0.2989 * Red[x][y] + 0.5871 * Green[x][y] + 0.1140 * Blue[x][y]
convertToGray(img,output_name)
#plotThresh(img)
