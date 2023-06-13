##############################
########Tugberk Dikmen########
###########21802480###########
###########CS484 HW2##########
##############################
import os
import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

def load_images(dir_path):
    
    #Load all images 
    images = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.png'):
            image = cv2.imread(file_path)
            images.append(image)
    return images

def convert_to_grayscale(image):
    
    #Convert to grayscale. 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def detect_edges(image, low_threshold, high_threshold):
    
    #Detect edges Canny edge 
    gray = convert_to_grayscale(image)
    
    #Canny edge detector
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def compute_histogram(edge_image, num_bins=8):
    
    #Compute line orientation histograms   
    dx, dy = np.gradient(edge_image)
    grad_mags = np.sqrt(dx ** 2 + dy ** 2)
    grad_orientations = np.arctan2(dy, dx)
    
    # Map orientations
    grad_orientations += math.pi
    grad_orientations %= math.pi
    
    # Compute histogram of orientations
    bin_size = math.pi / num_bins
    hist = np.zeros(num_bins)
    
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if edge_image[i, j] > 0:
                bin_idx = int(grad_orientations[i, j] // bin_size)
                hist[bin_idx] += grad_mags[i, j]
    
    hist /= np.sum(hist)   
    return hist

def find_match(rotated_hist, original_hists):
    min_distance = float('inf')
    best_match = None

    for i, hist in enumerate(original_hists):
        for shift in range(len(hist)):
            shifted_hist = np.roll(rotated_hist, shift)
            distance = np.linalg.norm(hist - shifted_hist)

            if distance < min_distance:
                min_distance = distance
                best_match = i

    return best_match

def find_rotation_angle(rotated_hist, original_hist, num_bins):
    
    #Find the angle of rotation with histo
    # Find the shift 
    distances = []
    for shift in range(len(original_hist)):
        shifted_hist = np.roll(rotated_hist, shift)
        distance = np.linalg.norm(shifted_hist - original_hist)
        distances.append(distance)
    min_distance_idx = np.argmin(distances)
    
    # Compute rotation angle
    bin_size = math.pi / num_bins
    rotation_angle = (min_distance_idx * bin_size * 180) / math.pi
    
    return rotation_angle

def find_lines(image, low_threshold, high_threshold):
    
    #Find straight line segments Hough transform lines
    gray = convert_to_grayscale(image)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
    
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return color_image

def main():
    #Load images
    original_dir = "template images"
    rotated_dir = "rotated images"

    original_images = load_images(original_dir)
    rotated_images = load_images(rotated_dir)
    line_images = load_images(original_dir)

    all_images = original_images + rotated_images
    
    low_threshold = 100
    high_threshold = 200
    
    edge_images = []
    
    print("--Show Edge Detected Images-- \n")

    for image in all_images:
        edge_image = detect_edges(image, low_threshold, high_threshold)
        edge_images.append(edge_image)
        pil_image = Image.fromarray(edge_image)
        pil_image.show()
    
    #overlay lines on original images
    print("--Show Overlaying Lines on Original Images-- \n")
   
    for image in line_images:
        image_with_lines = find_lines(image, low_threshold, high_threshold)

        cv2.imshow("Detected Lines", image_with_lines)
        cv2.waitKey(0)

    #Compute histograms for each image    
    print("--Show Oriantation Histograms of Images-- \n")

    num_bins = 8
    histograms = []

    for edge_image in edge_images:
        hist = compute_histogram(edge_image, num_bins=num_bins)
        histograms.append(hist)
       
    for i in range(5):
        histos = histograms[i]
        num_bins = len(histos)
        bin_size = math.pi / num_bins
        bins = np.arange(0, math.pi + bin_size, bin_size)
        plt.bar(bins[:-1], histos, width=bin_size)
        plt.xlabel('Orientation (radians)')
        plt.ylabel('Frequency')
        plt.title('Line Orientation Histogram')
        plt.show()
    
    #find the matching images using parts above
    print("--Find Matching Images-- \n")

    half_hists = len(histograms) // 2
    rotated_hists = histograms[half_hists:]
    original_hists = histograms[:half_hists]

    matches = []
    angles = []

    for i, rotated_hist in enumerate(rotated_hists):
        match_idx = find_match(rotated_hist, original_hists)
        matches.append(match_idx)
        
        original_hist = original_hists[match_idx]
        rotation_angle = find_rotation_angle(rotated_hist, original_hist, num_bins)
        angles.append(rotation_angle)


    for i in range (15):
        print("Match found for ", matches[i] +1 , ". original image as ", i +1, ". rotated image with the rotation angle ", angles[i], "\n")
    
main()