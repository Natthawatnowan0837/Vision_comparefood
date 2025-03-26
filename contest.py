import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set path
DATA_PATH = r"/home/jonut/compare_food/Test Set Samples/Test Images"
CSV_PATH = r"/home/jonut/compare_food/Test Set Samples/test.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Load model
loaded_model = tf.keras.models.load_model('/home/jonut/compare_food/trained_model_with_composition_and_blur.h5')

# Function to compute freshness score
def compute_freshness(image):
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue_mean = np.mean(hsv[:, :, 0]) / 180.0  # Normalize Hue (0-180 in OpenCV)
    sat_mean = np.mean(hsv[:, :, 1]) / 255.0  # Normalize Saturation (0-255)
    return np.array([hue_mean, sat_mean])

# Function to compute blur score (Laplacian Variance)
def compute_blur_score(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return np.array([laplacian_var / 1000])  # Normalize the score

# Function to compute composition score (Edge Density using Canny)
def compute_composition_score(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Canny edge detection
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])  # Calculate edge density
    return np.array([edge_density])

# Function to load and preprocess images
def load_and_preprocess_image(image_name):
    img_path = os.path.join(DATA_PATH, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Image {image_name} not found at {img_path}. Skipping...")
        return None, None, None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    
    freshness = compute_freshness(img)
    blur = compute_blur_score(img)
    composition = compute_composition_score(img)
    
    return img, freshness, blur, composition

# Predict images from CSV
def predict_images_from_csv():
    for index, row in df.iterrows():
        img1 = load_and_preprocess_image(row['Image 1'])
        img2 = load_and_preprocess_image(row['Image 2'])

        if img1[0] is None or img2[0] is None:
            continue  # Skip if image loading failed
        
        # Prepare inputs
        img1_data = np.expand_dims(img1[0], axis=0)  # Image 1
        img2_data = np.expand_dims(img2[0], axis=0)  # Image 2
        freshness1_data = np.expand_dims(img1[1], axis=0)  # Freshness 1
        freshness2_data = np.expand_dims(img2[1], axis=0)  # Freshness 2
        blur1_data = np.expand_dims(img1[2], axis=0)  # Blur 1
        blur2_data = np.expand_dims(img2[2], axis=0)  # Blur 2
        composition1_data = np.expand_dims(img1[3], axis=0)  # Composition 1
        composition2_data = np.expand_dims(img2[3], axis=0)  # Composition 2

        # Predict using the model
        prediction = loaded_model.predict([img1_data, img2_data, freshness1_data, freshness2_data, blur1_data, blur2_data, composition1_data, composition2_data])

        # Output the prediction (1 or 2 for image 1 or image 2)
        winner = 2 if prediction[0][0] > 0.5 else 1
        print(f"[{index+1}] Image {winner} is more delicious")

        # Show the images with the result
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img1[0])
        ax[0].set_title(f"Image 1\n{'✅' if winner == 1 else ''}")
        ax[0].axis("off")

        ax[1].imshow(img2[0])
        ax[1].set_title(f"Image 2\n{'✅' if winner == 2 else ''}")
        ax[1].axis("off")

        plt.suptitle(f"Prediction: Image {winner} is more delicious")
        plt.show()

# Call the function
predict_images_from_csv()
