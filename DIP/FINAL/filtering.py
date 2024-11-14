import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from skimage.util import img_as_float, img_as_ubyte
import cv2
import time  # Import time module for timing

def apply_filters_on_image(image_path, apply_on_superpixels=True):
    # Read the image using skimage
    image = io.imread(image_path)
    image_float = img_as_float(image)
    
    # Check if we need to apply SLIC segmentation
    start_time = time.time()  # Start timing
    if apply_on_superpixels:
        print("Applying SLIC superpixel segmentation...")
        # SLIC segmentation parameters
        slic_segments = 5000  # Adjust as needed
        slic_compactness = 50  # Adjust as needed
        
        # Apply SLIC segmentation
        slic_labels = segmentation.slic(image_float, n_segments=slic_segments, compactness=slic_compactness, start_label=1)
        
        # Create superpixel image by averaging colors within segments
        processed_image = color.label2rgb(slic_labels, image_float, kind='avg')
    else:
        print("Using the original image...")
        processed_image = image_float
    
    # Convert processed image to uint8 for OpenCV
    processed_image_uint8 = img_as_ubyte(processed_image)
    
    # Convert from RGB to BGR format for OpenCV
    processed_image_bgr = cv2.cvtColor(processed_image_uint8, cv2.COLOR_RGB2BGR)
    
    # Define filters with adjustable parameters
    filters = {
        'Median': {
            'function': lambda img, params: cv2.medianBlur(img, **params),
            'params': {'ksize': 5},
            'expects': 'either',
            'returns': 'same',
        },
        'Gaussian': {
            'function': lambda img, params: cv2.GaussianBlur(img, **params),
            'params': {'ksize': (5, 5), 'sigmaX': 2},
            'expects': 'either',
            'returns': 'same',
        },
        'Bilateral': {
            'function': lambda img, params: cv2.bilateralFilter(img, **params),
            'params': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'expects': 'color',
            'returns': 'color',
        },
        'Grayscale': {
            'function': lambda img, params: cv2.cvtColor(img, **params),
            'params': {'code': cv2.COLOR_BGR2GRAY},
            'expects': 'color',
            'returns': 'gray',
        },
        'KNN_Color': {
            'function': lambda img, params: cv2.fastNlMeansDenoisingColored(img, **params),
            'params': {'h': 10, 'hColor': 10, 'templateWindowSize': 7, 'searchWindowSize': 21},
            'expects': 'color',
            'returns': 'color',
        },
        'KNN_Gray': {
            'function': lambda img, params: cv2.fastNlMeansDenoising(img, **params),
            'params': {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21},
            'expects': 'gray',
            'returns': 'gray',
        },
    }
    
    # Define sequences of filters to apply
    sequences_to_apply = [
        ['Median'],
        ['Gaussian'],
        ['Bilateral'],
        ['Grayscale'],
        ['KNN_Color'],
        ['KNN_Gray'],
        ['Grayscale', 'KNN_Gray'],
        ['Gaussian', 'KNN_Color'],
        ['Median', 'KNN_Color'],
        ['Bilateral', 'Grayscale', 'KNN_Gray'],
        # Add more sequences as desired
    ]
    
    # Apply each sequence to the processed image
    for sequence in sequences_to_apply:
        print(f"Applying filter sequence: {' -> '.join(sequence)}")
        
        image = processed_image_bgr.copy()
        image_type = 'color'  # Initial image is color
        
        for filter_name in sequence:
            filter_info = filters[filter_name]
            function = filter_info['function']
            params = filter_info['params']
            expects = filter_info['expects']
            returns = filter_info['returns']
            
            # Check if image type matches filter expectation
            if expects == 'either' or expects == image_type:
                pass  # No conversion needed
            elif expects == 'color' and image_type == 'gray':
                # Convert grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_type = 'color'
            elif expects == 'gray' and image_type == 'color':
                # Convert BGR to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_type = 'gray'
            else:
                raise ValueError(f"Unsupported image type conversion from {image_type} to {expects}")
            
            # Apply the filter
            image = function(image, params)
            
            # Update image_type
            if returns == 'same':
                pass  # image_type remains the same
            else:
                image_type = returns
        
        # Prepare image for saving
        if image_type == 'color':
            # Convert image from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cmap = None
        elif image_type == 'gray':
            # Image is grayscale
            image_rgb = image
            cmap = 'gray'
        else:
            raise ValueError(f"Unsupported image type: {image_type}")
        
        # Save the image
        plt.figure()
        plt.imshow(image_rgb, cmap=cmap)
        plt.axis('off')
        
        # Create filename based on whether filters were applied on superpixels or original image
        sequence_name = '_'.join(sequence)
        base_filename = f'filtered_{sequence_name}'
        if apply_on_superpixels:
            filename = f'./slic_image/{base_filename}_superpixels.png'
        else:
            filename = f'./processed_images/{base_filename}_original.png'
        
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Filtering complete. Images saved. Time taken: {elapsed_time:.2f} seconds.")

# Example usage
# To apply filters on the superpixel image:
apply_filters_on_image('greenCloth.jpeg', apply_on_superpixels=True)

# To apply filters on the original image:
apply_filters_on_image('greenCloth.jpeg', apply_on_superpixels=False)
