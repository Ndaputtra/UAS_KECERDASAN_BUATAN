import cv2
import os
import numpy as np
from PIL import Image

def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        pil_img = Image.open(image_path).convert('L')  # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split("_")[1].split(".")[0])
        face_samples.append(img_numpy)
        ids.append(id)
    return face_samples, ids

dataset_path = 'dataset'
faces, ids = get_images_and_labels(dataset_path)

print(f"Found {len(faces)} faces")

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    recognizer = cv2.face.createLBPHFaceRecognizer()

recognizer.train(faces, np.array(ids))

model_path = 'trainer/trainer.yml'
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.save(model_path)

print(f"[INFO] {len(np.unique(ids))} faces trained. Model saved at {model_path}")
