import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Set path
DATA_PATH = r"D:\Code\compare_food\Questionair Images"
CSV_PATH = r"D:\Code\compare_food\data_from_questionaire.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

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

# Image processing function
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

# Load images with filtering out None values
data = [load_and_preprocess_image(img) for img in df['Image 1']]
X1 = np.array([d[0] for d in data if d[0] is not None])
F1 = np.array([d[1] for d in data if d[1] is not None])
B1 = np.array([d[2] for d in data if d[2] is not None])
C1 = np.array([d[3] for d in data if d[3] is not None])

data = [load_and_preprocess_image(img) for img in df['Image 2']]
X2 = np.array([d[0] for d in data if d[0] is not None])
F2 = np.array([d[1] for d in data if d[1] is not None])
B2 = np.array([d[2] for d in data if d[2] is not None])
C2 = np.array([d[3] for d in data if d[3] is not None])

y_classification = np.array(df['Winner']) - 1  # Convert 1,2 to 0,1

# Ensure dataset consistency
min_size = min(len(X1), len(X2), len(y_classification))
X1, X2, F1, F2, B1, B2, C1, C2, y_classification = X1[:min_size], X2[:min_size], F1[:min_size], F2[:min_size], B1[:min_size], B2[:min_size], C1[:min_size], C2[:min_size], y_classification[:min_size]

# Split dataset
X1_train, X1_test, X2_train, X2_test, F1_train, F1_test, F2_train, F2_test, B1_train, B1_test, B2_train, B2_test, C1_train, C1_test, C2_train, C2_test, y_class_train, y_class_test = train_test_split(
    X1, X2, F1, F2, B1, B2, C1, C2, y_classification, test_size=0.2, random_state=42)

# Define CNN base model
def create_base_cnn():
    input_layer = Input(shape=(224, 224, 3), name='input_layer')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)  # เพิ่ม Dropout ตรงนี้
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)  # เพิ่ม Dropout ตรงนี้
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)  # เพิ่ม Dropout ตรงนี้
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)  # เพิ่ม Dropout ตรงนี้
    return Model(input_layer, x)

# Create twin networks
base_cnn = create_base_cnn()
input_1 = Input(shape=(224, 224, 3), name='input_1')
input_2 = Input(shape=(224, 224, 3), name='input_2')
freshness_1 = Input(shape=(2,), name='freshness_1')
freshness_2 = Input(shape=(2,), name='freshness_2')
blur_1 = Input(shape=(1,), name='blur_1')
blur_2 = Input(shape=(1,), name='blur_2')
composition_1 = Input(shape=(1,), name='composition_1')
composition_2 = Input(shape=(1,), name='composition_2')

encoded_1 = base_cnn(input_1)
encoded_2 = base_cnn(input_2)

merged = Concatenate()([encoded_1, encoded_2, freshness_1, freshness_2, blur_1, blur_2, composition_1, composition_2])

# Classification head
class_output = Dense(1, activation='sigmoid', name='class_output')(merged)

# Build model
model = Model(inputs=[input_1, input_2, freshness_1, freshness_2, blur_1, blur_2, composition_1, composition_2], 
              outputs=class_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit([X1_train, X2_train, F1_train, F2_train, B1_train, B2_train, C1_train, C2_train], y_class_train,
          validation_data=([X1_test, X2_test, F1_test, F2_test, B1_test, B2_test, C1_test, C2_test], y_class_test),
          epochs=100, batch_size=64)

# Save model
model.save('trained_model_with_composition_and_blur.h5')

# Evaluate model
loss, accuracy = model.evaluate([X1_test, X2_test, F1_test, F2_test, B1_test, B2_test, C1_test, C2_test], y_class_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
