# histogram.py
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
def plot_rgb_channels(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
    
        r, g, b = cv2.split(img)
        
    
        zeros = np.zeros_like(r)
        
       
        red_img = cv2.merge([r, zeros, zeros])     
        green_img = cv2.merge([zeros, g, zeros])   
        blue_img = cv2.merge([zeros, zeros, b])    
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(red_img)
        axs[0].set_title('Red Channel')
        axs[0].axis('off')
        
        axs[1].imshow(green_img)
        axs[1].set_title('Green Channel')
        axs[1].axis('off')
        
        axs[2].imshow(blue_img)
        axs[2].set_title('Blue Channel')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def compute_histogram(img, bins, mask=None):
   #working for rgb or gray
   # does mask or no maske based on if mask was supplied
   # aka regional vs entire image
   # gray very easy rgb a bit hectic
    if len(img.shape) == 2: 
        #do a ravel(like a flatten but no py copy :) ) using numpy so we can iterate over all pixels without having to manage several iterations, we just want pixels...
        #other way is to iterate over the img data 2d array and use [x,y] reference which feels harder
        data = img[mask == 1] if mask is not None else img.ravel()
        hist = [0] * bins
        binSize = 256 / bins
        for pixel in data:
            binIDx = int(pixel / binSize)
            if binIDx >= bins:
                binIDx = bins - 1
            hist[binIDx] += 1
        return {'gray': hist}
    if len(img.shape) == 3 and img.shape[2] == 3: 
        histograms = {}
        channels = ('r', 'g', 'b') 
        binSize = 256 / bins
        for i, color in enumerate(channels):
            channel_data = img[:, :, i]
            #here we again do our ravel so we get to work with the pixels, but we do this after our slice of each channel 
            #again we ravel so we can just iterate over pixels since we do not care about the location
            # the mask will handle location math and only return pixels we want to iterate 
            if mask is not None:
                channel_data = channel_data[mask == 1]
            hist = [0] * bins
            for pixel in channel_data.ravel():
                #do some int math to fit the bin
                binIDx = int(pixel / binSize)
                if binIDx >= bins:
                    binIDx = bins - 1
                hist[binIDx] += 1
            histograms[color] = hist
        return histograms

def plot_histogram(hist, bins, title='Image Histogram'):
    binEdge = np.arange(bins)
    
    if 'gray' in hist:
        # Grayscale image
        plt.figure(figsize=(10, 6))
        plt.bar(binEdge, hist['gray'], color='gray', width=1)
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, bins-1])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    else:
        # overlap plot for easy scan
        plt.figure(figsize=(10, 6))
        colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
        for color, values in hist.items():
            plt.bar(binEdge, values, color=colors[color], alpha=0.5, label=f'{color.upper()} channel', width=1)
        plt.legend()
        plt.title(f'{title} (Combined RGB)')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, bins-1])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # each channel histo for assignment
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        for i, (color, values) in enumerate(hist.items()):
            axs[i].bar(binEdge, values, color=colors[color], width=1)
            axs[i].set_title(f'{color.upper()} Channel Histogram')
            axs[i].set_xlabel('Pixel Intensity')
            axs[i].set_ylabel('Frequency')
            axs[i].set_xlim([0, bins-1])
            axs[i].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

def preprocess_mask(mask):
   
    # needed to do this since values where not binary one time, think i did somethin lol
    _, BMask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return BMask

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python histogram.py <image> <bins> [mask]")
        sys.exit(1)
    
    IPath = sys.argv[1]
    bins = int(sys.argv[2])
    mask_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Load image
    if mask_path:
        # read in png as default cv
        img = cv2.imread(IPath, cv2.IMREAD_UNCHANGED)
        #now do our alpha check which I think png should have
        if len(img.shape) == 3 and img.shape[2] == 4:
            # convert standard cv img into the color img
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            #do same thing again but this time do not need to account for alpha channel 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            #grayscale ez
            pass
        else:
            sys.exit(1)
        
        maskImg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = preprocess_mask(maskImg)
    else:
        #same code as above with no mask stuff
        #could save code time but was changing as went
        img = cv2.imread(IPath, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
           
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            
            pass
        else:
            sys.exit(1)
        mask = None
        
    start_time = time.perf_counter()
    hist = compute_histogram(img, bins, mask)
    end_time = time.perf_counter() 
    execution_time = end_time - start_time
    plot_rgb_channels(img)
    print(f"Histogram time: {execution_time} seconds")
    plot_histogram(hist, bins, title='Histogram')
    
