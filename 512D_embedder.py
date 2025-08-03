#512D_embedder.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from keras_facenet import FaceNet
import json

# Paths
CROPPED_DIR = 'images/cropped'
EMBEDDING_DIR = 'embeddings'

# Initialize FaceNet model
embedder = FaceNet()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Processing image: {image_path} with shape {img.shape} and min/max pixel values {img.min()}/{img.max()}")
    return img

def generate_embedding(img_path):
    img = preprocess_image(img_path)
    embedding = embedder.embeddings([img])[0]  # shape: (512,)
    return embedding

def save_embedding(person_id, image_name, embedding):
    person_dir = os.path.join(EMBEDDING_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    # Convert to list and save as JSON
    embedding_list = embedding.tolist()
    json_path = os.path.join(person_dir, image_name.replace('.jpg', '.json').replace('.png', '.json'))
    with open(json_path, 'w') as f:
        json.dump(embedding_list, f)

def main():
    persons = [p for p in os.listdir(CROPPED_DIR) if os.path.isdir(os.path.join(CROPPED_DIR, p))]

    for person in tqdm(persons, desc="Generating embeddings"):
        person_path = os.path.join(CROPPED_DIR, person)
        images = [img for img in os.listdir(person_path) if img.lower().endswith(('.jpg', '.png'))]

        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            embedding = generate_embedding(img_path)
            save_embedding(person, img_file, embedding)

if __name__ == '__main__':
    main()
