#matcher.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.distance import hamming
import glob
from facenet_embedder import embedder

# Import your alignment + cropping function from preprocessing module
from MTCNN_preprocess_gpu import align_and_crop_face

# Parameters
CLASS_HV_DIR = 'class_hypervectors'
HV_DIM = 10000
projection_matrix = np.random.RandomState(42).randn(512, HV_DIM)

def preprocess_image(image_path):
    face = align_and_crop_face(image_path)  # Must return RGB 160x160 numpy array or None
    if face is None:
        raise ValueError(f"Face alignment and cropping failed for {image_path}")
    return face

def get_embedding(image_path):
    img = preprocess_image(image_path)
    return embedder.embeddings([img])[0]

def encode_embedding_to_hv(embedding):
    projected = np.dot(embedding, projection_matrix)
    return (projected > 0).astype(np.uint8)

def load_class_hypervectors():
    class_hv_list = []
    person_ids = []

    for file in os.listdir(CLASS_HV_DIR):
        if file.endswith('_class_hv.json'):
            person_id = file.replace('_class_hv.json', '')
            with open(os.path.join(CLASS_HV_DIR, file), 'r') as f:
                hv = np.array(json.load(f), dtype=np.uint8)
                packed = np.packbits(hv)  # shape: (HV_DIM // 8,)
                class_hv_list.append(packed)
                person_ids.append(person_id)

    class_hv_matrix = np.stack(class_hv_list)  # shape: (num_classes, HV_DIM // 8)
    return person_ids, class_hv_matrix

# Load class hypervectors once globally
person_ids, class_hv_matrix = load_class_hypervectors()

def match_hv_to_class(query_hv, class_hv_matrix, person_ids):
    try:
        if len(class_hv_matrix) == 0 or len(person_ids) == 0:
            raise ValueError("No class hypervectors or labels provided.")

        query_packed = np.packbits(query_hv.astype(np.uint8))
        xor_result = np.bitwise_xor(class_hv_matrix, query_packed)
        hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1) / HV_DIM
        similarities = 1 - hamming_dist

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_label = person_ids[best_idx]

        return best_label, best_score
    except Exception as e:
        print(f"❌ Error in match_hv_to_class: {e}")
        return None, None

def classify(query_image_path, person_ids, class_hv_matrix, top_k=1):
    try:
        embedding = get_embedding(query_image_path)
        query_hv = encode_embedding_to_hv(embedding)
        query_packed = np.packbits(query_hv)  # shape: (HV_DIM // 8,)

        # Vectorized XOR
        xor_result = np.bitwise_xor(class_hv_matrix, query_packed)  # shape: (num_classes, HV_DIM // 8)

        # Count differing bits using unpackbits
        hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1) / HV_DIM  # normalized distance
        similarities = 1 - hamming_dist

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(person_ids[i], similarities[i]) for i in top_indices]

    except Exception as e:
        print(f"❌ Error processing {query_image_path}: {e}")
        return []

def main():
    person_ids, class_hv_matrix = load_class_hypervectors()

    test_images = glob.glob('images/test_faces/*.jpg') + \
                  glob.glob('images/test_faces/*.png') + \
                  glob.glob('images/test_faces/*.webp')

    for image_path in tqdm(test_images, desc="Classifying test images"):
        top_matches = classify(image_path, person_ids, class_hv_matrix, top_k=2)
        print(f"\n📷 Query: {os.path.basename(image_path)}")
        for person_id, score in top_matches:
            print(f" → Match: {person_id}, Similarity: {score:.4f}")


if __name__ == '__main__':
    main()
