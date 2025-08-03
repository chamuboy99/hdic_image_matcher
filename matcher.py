#matcher.py

import os
import numpy as np
import json
from keras_facenet import FaceNet
from tqdm import tqdm
from scipy.spatial.distance import hamming
import glob

# Import your alignment + cropping function from preprocessing module
from MTCNN_preprocess_gpu import align_and_crop_face

# Parameters
CLASS_HV_DIR = 'class_hypervectors'
HV_DIM = 10000
projection_matrix = np.random.RandomState(42).randn(512, HV_DIM)

# Initialize FaceNet embedder once
embedder = FaceNet()

def preprocess_image(image_path):
    """
    Preprocess image by performing alignment and cropping using imported function.
    Returns aligned RGB face resized to 160x160.
    """
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
        top_matches = classify(image_path, person_ids, class_hv_matrix, top_k=3)
        print(f"\n📷 Query: {os.path.basename(image_path)}")
        for person_id, score in top_matches:
            print(f" → Match: {person_id}, Similarity: {score:.4f}")


if __name__ == '__main__':
    main()
