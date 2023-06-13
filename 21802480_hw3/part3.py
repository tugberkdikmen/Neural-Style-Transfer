import numpy as np
import cv2
from sklearn.cluster import KMeans

#Paths
features_path = 'features output'
images_path = 'input'
output_path_labels = 'clusters output w labels'
output_path = 'clusters output'
labels_path = 'label output'

# Load the superpixel features
features = []
for i in range(1, 11):  
    feature_path = f'{features_path}/features{i}.npy'  
    features.append(np.load(feature_path))
features = np.concatenate(features, axis=0)

# Clustering
num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters)
labels = kmeans.fit_predict(features)

# Generate pseudo-color representation for each image
for i in range(1, 11):  
    image_path = f'{images_path}/{i}.jpg'  # image
    label_path = f'{labels_path}/label{i}.png' # label image
    output_image_path = f'{output_path}/image{i}_clustered.png'  # output image
    output_image_path_labels = f'{output_path_labels}/image{i}_clustered_labels.png'  # output image
   
    # Load orj image
    image = cv2.imread(image_path)

    # Resize the labels
    resized_labels = cv2.resize(labels.reshape(-1, 1), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Assign pseudo-colors to pixels based on the cluster labels
    colors = [np.random.randint(0, 255, 3) for _ in range(num_clusters)]
    for label, color in zip(np.unique(resized_labels), colors):
        mask = resized_labels == label  #mask for pixels belonging to the current cluster
        image[mask] = color

    cv2.imwrite(output_image_path, image)

    # Load label image
    labelImages = cv2.imread(label_path)

    # Resize the labels 
    resized_labels = cv2.resize(labels.reshape(-1, 1), (labelImages.shape[1], labelImages.shape[0]), interpolation=cv2.INTER_NEAREST)

    #pseudo-colors to pixels based on the cluster labels
    colors = [np.random.randint(0, 255, 3) for _ in range(num_clusters)]
    for label, color in zip(np.unique(resized_labels), colors):
        mask = resized_labels == label  # Create a boolean mask for pixels belonging to the current cluster
        labelImages[mask] = color

    cv2.imwrite(output_image_path_labels, labelImages)