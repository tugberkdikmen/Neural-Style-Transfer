import cv2
import numpy as np
from skimage.segmentation import slic

def obtain_superpixels(image, num_segments, compactness):
    #Convert image to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    #SLIC superpixel algorithm
    labels = slic(image, n_segments=num_segments, compactness=compactness)

    return labels

def save_label_image(labels, output_path):
    #conv grayscale
    cv2.imwrite(output_path, labels.astype(np.uint8))

# Parameters for SLIC superpixels
num_segments = 1500  #desired superpixels
compactness = 10  #compactness

#the dataset
dataset_path = 'input'
output_path = 'label output'

for i in range(1, 11):  
    image_path = f'{dataset_path}/{i}.jpg'  #image file 
    output_image_path = f'{output_path}/label{i}.png'  #output file 
    try:
        image = cv2.imread(image_path)

        if image is None:
            raise Exception(f"Failed to load image: {image_path}")

        # Obtain superpixels
        labels = obtain_superpixels(image, num_segments, compactness)

        # Save image
        save_label_image(labels, output_image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")