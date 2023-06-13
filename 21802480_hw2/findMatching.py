import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

def canny_edge_detection(img_gray, threshold1, threshold2):
    # Convert im age to grayscale
    edges = cv2.Canny(img_gray, threshold1, threshold2)
    return edges

def line_fitting(edges, rho_resolution, theta_resolution, threshold):
    # Apply Hough transform to detect lines
    lines = cv2.HoughLines(edges, rho_resolution, theta_resolution, threshold)

    # Extract line segments
    line_segments = []
    if lines is not None:
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_segments.append(((x1, y1), (x2, y2)))
    
    return line_segments

def compute_orientation_histogram(line_segments, num_bins):
    # Compute the orientation histogram
    orientation_histogram = np.zeros(num_bins)
    for i in range(len(line_segments)):
        x1, y1 = line_segments[i][0]
        x2, y2 = line_segments[i][1]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)
        bin_index = int((angle + np.pi) / (2 * np.pi) * num_bins)
        orientation_histogram[bin_index] += np.sqrt(dx**2 + dy**2)
    
    return orientation_histogram

def find_rotation_angle(rotated_histogram, original_histogram):
    # Find the rotation angle based on histogram similarity
    num_bins = len(rotated_histogram)
    min_distance = float('inf')
    best_shift = 0
    for shift in range(num_bins):
        distance = np.linalg.norm(np.roll(rotated_histogram, shift) - original_histogram)
        if distance < min_distance:
            min_distance = distance
            best_shift = shift
    rotation_angle = best_shift * (2 * np.pi / num_bins)
    
    return rotation_angle

def find_matching_book(rotated_histograms, original_histograms):
    # Find the original book for each rotated book
    num_books = len(rotated_histograms)
    matchings = []
    for i in range(num_books):
        best_distance = float('inf')
        best_index = -1
        for j in range(num_books):
            distance = np.linalg.norm(np.roll(rotated_histograms[i], -j) - original_histograms[j])
            if distance < best_distance:
                best_distance = distance
                best_index = j
        matchings.append(best_index)
    return matchings

def load_images(dir_path):
  
    # Get all file names in the directory
    files = os.listdir(dir_path)

    # Load all image files
    images = []
    for file_name in files:
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
            file_path = os.path.join(dir_path, file_name)
            image = Image.open(file_path)
            images.append(image)
    
    return images

def preprocess_image(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    return edges

def detect_lines(image):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines

def compute_histogram(lines, num_bins):
    # compute line orientations and lengths
    orientations = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
    lengths = np.sqrt((lines[:, 3] - lines[:, 1])**2 + (lines[:, 2] - lines[:, 0])**2)
    # compute weighted histogram of line orientations
    hist, _ = np.histogram(orientations, bins=num_bins, weights=lengths)
    return hist

def match_rotation(rotated_hist, original_hists):
    min_dist = float('inf')
    best_match = None
    for i, original_hist in enumerate(original_hists):
        for shift in range(len(original_hist)):
            shifted_rotated_hist = np.roll(rotated_hist, shift)
            dist = np.linalg.norm(original_hist - shifted_rotated_hist)
            if dist < min_dist:
                min_dist = dist
                best_match = i
    # estimate angle of rotation
    angle = best_match * (2*np.pi/len(original_hists))
    return angle

def main():
    print("--Find Matching--")
    # load images
    original_images = load_images("template images")
    rotated_images = load_images("rotated images")
    # preprocess images and detect edges and lines
    original_edges = [detect_edges(preprocess_image(image)) for image in original_images]
    rotated_edges = [detect_edges(preprocess_image(image)) for image in rotated_images]
    original_lines = [detect_lines(edges) for edges in original_edges]
    rotated_lines = [detect_lines(edges) for edges in rotated_edges]
    # compute histograms of line orientations
    num_bins = 36
    original_hists = [compute_histogram(lines, num_bins) for lines in original_lines]
    rotated_hists = [compute_histogram(lines, num_bins) for lines in rotated_lines]
    # match rotation of each rotated book to its original book
    for i, rotated_hist in enumerate(rotated_hists):
        angle = match_rotation(rotated_hist, original_hists)
        print(f"Rotated image {i} is rotated by {angle*180/np.pi} degrees")
    
main()