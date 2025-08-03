#class_hypervector_generation.py

import os
import json
import numpy as np
from tqdm import tqdm

# Parameters
HYPERVECTOR_DIR = 'hypervectors'        # Per-image hypervectors
CLASS_HV_DIR = 'class_hypervectors'     # One class HV per person
HV_DIM = 10000

def load_hypervector(path):
    with open(path, 'r') as f:
        return np.array(json.load(f), dtype=np.uint8)

def create_class_hypervector(person_id):
    person_dir = os.path.join(HYPERVECTOR_DIR, person_id)
    hv_files = [f for f in os.listdir(person_dir) if f.endswith('_hv.json')]

    if not hv_files:
        return None

    # Load and stack all image HVs
    hypervectors = [load_hypervector(os.path.join(person_dir, f)) for f in hv_files]
    hypervectors = np.stack(hypervectors)  # shape: (num_images, HV_DIM)

    # Sum element-wise
    summed = np.sum(hypervectors, axis=0)  # shape: (HV_DIM,)

    # Threshold to binary: majority vote
    threshold = len(hypervectors) / 2
    class_hv = (summed > threshold).astype(np.uint8)

    return class_hv

def save_class_hypervector(person_id, hv):
    os.makedirs(CLASS_HV_DIR, exist_ok=True)
    out_path = os.path.join(CLASS_HV_DIR, f"{person_id}_class_hv.json")
    with open(out_path, 'w') as f:
        json.dump(hv.tolist(), f)

def main():
    persons = [d for d in os.listdir(HYPERVECTOR_DIR) if os.path.isdir(os.path.join(HYPERVECTOR_DIR, d))]

    for person in tqdm(persons, desc="Generating class hypervectors"):
        class_hv = create_class_hypervector(person)
        if class_hv is not None:
            save_class_hypervector(person, class_hv)

if __name__ == '__main__':
    main()
