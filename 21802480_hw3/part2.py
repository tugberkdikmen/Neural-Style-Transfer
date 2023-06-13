import cv2
import numpy as np
from skimage.util import img_as_float
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def gaborconvolve(image, filters):
    # Preprocess image
    image = img_as_float(image)

    # Compute Gabor filter responses
    responses = []
    for kernel in filters:
        response = convolve2d(image, kernel, mode='same')
        responses.append(response)

    return responses

def compute_gabor_features(image, labels, filters):
    num_superpixels = np.max(labels) + 1
    num_filters = len(filters)

    # Compute Gabor filter responses for each pixel
    responses = gaborconvolve(image, filters)
    
    # Compute Gabor features for each superpixel
    features = np.zeros((num_superpixels, num_filters))
    for i in range(num_superpixels):
        indices = np.where(labels == i)
        for j in range(num_filters):
            response = responses[j]
            feature = np.mean(response[indices])
            features[i, j] = feature

    return features

# Parameters for Gabor texture features
scales = [1, 2, 4, 8]  # Number of scales
orientations = [0, 45, 90, 135]  # Orientations in degrees

# Generate Gabor filters
filters = []
for scale in scales:
    for orientation in orientations:
        wavelength = 2.0 / scale
        theta = orientation * np.pi / 180.0
        kernel = cv2.getGaborKernel((0, 0), wavelength, theta, 10.0, 0.5, 0, ktype=cv2.CV_64F)
        filters.append(kernel)

# Process each image in the dataset
dataset_path = 'input'
label_path = 'label output'
output_path = 'features output'

for i in range(1, 11):  
    image_path = f'{dataset_path}/{i}.jpg'  # image file 
    label_image_path = f'{label_path}/label{i}.png'  # label image 
    output_feature_path = f'{output_path}/features{i}.npy'  # output file 
    output_feature_pathCSV = f'{output_path}/featuresCSV{i}.csv'  # output file 

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception(f"Failed to load image: {image_path}")

        labels = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)

        if labels is None:
            raise Exception(f"Failed to load label image: {label_image_path}")

        # Compute Gabor features for superpixels
        features = compute_gabor_features(image, labels, filters)

        # Save the features as numpy array
        np.save(output_feature_path, features)

        # Visualize features as an image
        #plt.imshow(features)
        #plt.colorbar()
        #plt.show()

        # Save features as CSV file
        #np.savetxt(output_feature_pathCSV, features, delimiter=',')

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")