import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to load images and labels
def load_dataset(dataset_dir):
    images = []
    labels = []
    for filename in os.listdir(dataset_dir):
        if filename.startswith('face_'):
            img_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Resize image to a consistent size
                image = cv2.resize(image, (100, 100))
                images.append(image)
                labels.append(int(filename.split('_')[1].split('.')[0]))  # Extract label from filename
    return images, labels

# Create a directory to store the dataset if it doesn't exist
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set the width
cam.set(4, 480)  # set the height

# Load the face detector
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Counter for the saved images
img_counter = 0

while True:
    retV, frame = cam.read()
    if not retV:
        break
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Save the detected face
        face_img = abuAbu[y:y+h, x:x+w]
        img_counter += 1
        img_name = f"{dataset_dir}/face_{img_counter}.jpg"
        cv2.imwrite(img_name, face_img)
    
    cv2.imshow('webcamku', frame)
    #cv2.imshow('webcam - Grey', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

# Release the camera correctly
cam.release()
cv2.destroyAllWindows()

# Load dataset
images, labels = load_dataset(dataset_dir)

# Convert images and labels to numpy arrays
images = np.array(images) / 255.0  # Normalize images
labels = np.array(labels)

# Split dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(100, 100)),  # Flatten the input (100x100 image)
    layers.Dense(128, activation='relu'),    # Dense layer with 128 neurons and ReLU activation
    layers.Dense(10)                         # Output layer with 10 neurons (assuming 10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
