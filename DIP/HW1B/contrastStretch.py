import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Function to perform linear contrast stretching
def linearStretch(img, newMin=0, newMax=255, oldMin=None, oldMax=None):
    if oldMin is None:
        oldMin = np.min(img)
    if oldMax is None:
        oldMax = np.max(img)
    stretched = (img.astype(np.float32) - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin
    stretched = np.clip(stretched, newMin, newMax).astype(np.uint8)
    return stretched

def plotImgHist(orig, enh, t1, t2):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # save my decrepid fingers from plotting
    def plot_image(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    # more plot code boring
    def plot_histogram(ax, img, title):
        ax.hist(img.flatten(), bins=256, range=[0, 256], color='gray')
        ax.set_title('Hist ' + title)

    plot_image(axs[0, 0], orig, t1)
    plot_image(axs[0, 1], enh, t2)
    plot_histogram(axs[1, 0], orig, t1)
    plot_histogram(axs[1, 1], enh, t2)

    plt.tight_layout()
    plt.show()

def computeLimits(img, discardPct):
    totalPix = img.size
    discardCnt = int(totalPix * discardPct / 100)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    #numpy is the best lol
    cumHist = np.cumsum(hist)
    #np thank you
    newMin = np.searchsorted(cumHist, discardCnt)
    #another np dub + list reversal for the sorted histogram now minus our discard limit
    newMax = 255 - np.searchsorted(cumHist[::-1], discardCnt)

    return newMin, newMax

# Main functionality
if __name__ == "__main__":
    
    imgPath = sys.argv[1]
    discardPcts = list(map(float, sys.argv[2:]))

    # Read the grayscale image
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    
    oldMin = np.min(img)
    oldMax = np.max(img)
    #simple stretch before the discard
    enhA = linearStretch(img, oldMin=oldMin, oldMax=oldMax)
    plotImgHist(img, enhA, 'input Img', 'Enhanced Img (Stretch)')

    for pct in discardPcts:
        newMin, newMax = computeLimits(img, discardPct=pct)
        enhB = linearStretch(img, oldMin=newMin, oldMax=newMax)
        plotImgHist(img, enhB, 'input Img', f'stretch Img ({pct}% Discard)')
