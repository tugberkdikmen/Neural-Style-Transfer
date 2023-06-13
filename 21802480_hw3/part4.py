import numpy as np
import cv2
from sklearn.cluster import KMeans

# Load Paths
features_path = 'features output'
images_path = 'input'
output_path = 'clusters final output'
labels_path = 'label output'

# Load superpixel features
features = []
for i in range(1, 11):  
    feature_path = f'{features_path}/features{i}.npy'  
    features.append(np.load(feature_path))
features = np.concatenate(features, axis=0)

# Clustering
num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
labels = kmeans.fit_predict(features)

# Generate pseudo-color representation for each image
for i in range(1, 11):  
    image_path = f'{images_path}/{i}.jpg'  # original image
    label_path = f'{labels_path}/label{i}.png'  # label image
    output_image_path = f'{output_path}/image{i}_clustered.png'  # output image

    # Load orj image
    image = cv2.imread(image_path)

    # Resize the labels 
    resized_labels = cv2.resize(labels.reshape(-1, 1), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Assign pseudo-colors to pixels based on the cluster labels
    colors = [np.random.randint(0, 255, 3) for _ in range(num_clusters)]
    for label, color in zip(np.unique(resized_labels), colors):
        mask = resized_labels == label  #mask for pixels belonging to the current cluster
        image[mask] = color

    # Save output
    cv2.imwrite(output_image_path, image)

    # first-level neighbors
    num_superpixels = labels.max() + 1
    num_features = features.shape[1]
    first_level_features = np.zeros((num_superpixels, num_features))
    for label in range(num_superpixels):
        mask = resized_labels == label  # mask for pixels belonging to the current superpixel
        indices = np.where(mask)[0]  # indices of the pixels belonging to the current superpixel
        if indices.size > 0:
            first_level_indices = np.unique(labels[indices])
            first_level_features[label] = np.mean(features[np.isin(labels, first_level_indices)], axis=0)

    # second-level neighbors
    second_level_features = np.zeros((num_superpixels, num_features))
    for label in range(num_superpixels):
        first_level_indices = np.unique(labels[np.where(resized_labels == label)[0]])
        second_level_indices = np.unique(labels[np.isin(labels, first_level_indices)])
        second_level_features[label] = np.mean(features[np.isin(labels, second_level_indices)], axis=0)

    # average feature vectors
    superpixel_features = features
    first_level_average_features = first_level_features[labels]
    second_level_average_features = second_level_features[labels]

    contextual_features = np.concatenate((superpixel_features, first_level_average_features, second_level_average_features), axis=1)

    # clustering contextual features
    contextual_num_clusters = 5  # Adjust the number of clusters for contextual features
    contextual_kmeans = KMeans(n_clusters=contextual_num_clusters)
    contextual_labels = contextual_kmeans.fit_predict(contextual_features)

    # pseudo-color representation based on the cluster labels
    contextual_output_image_path = f'{output_path}/image{i}_contextual_clustered.png'
    contextual_colors = [np.random.randint(0, 255, 3) for _ in range(contextual_num_clusters)]
    for label, color in zip(np.unique(contextual_labels), contextual_colors):
        mask = contextual_labels == label
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    image[np.where(mask_resized)] = color

    # Save contextual clustering
    cv2.imwrite(contextual_output_image_path, image)