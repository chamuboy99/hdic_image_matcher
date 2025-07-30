from pathlib import Path
from src.face_preprocess import detect_and_crop_face
from src.embedder2 import get_embedding
from src.matcher import load_database
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import cv2

def match_embedding_to_db(embedding, db, threshold=0.65):
    best_match = None
    best_score = -1

    for person in db:
        db_vector = np.array(person["embedding"])
        score = cosine_similarity([embedding], [db_vector])[0][0]

        if score > best_score:
            best_score = score
            best_match = person["person_id"]

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def batch_match(input_dir, db_path, temp_crop_dir="../images/cropped_inputs", threshold=0.65):
    input_dir = Path(input_dir)
    crop_dir = Path(temp_crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    db = load_database(db_path)

    print(f"\n🔍 Matching incoming faces from {input_dir}...\n")
    for img_path in input_dir.glob("*.jpg"):
        cropped_path = crop_dir / img_path.name

        # Step 1: Crop face
        face = detect_and_crop_face(str(img_path), str(cropped_path), save=True)
        if face is None:
            print(f"❌ No face detected in {img_path.name}")
            continue

        # Step 2: Embed and Match
        embedding = get_embedding(str(cropped_path))
        match_id, score = match_embedding_to_db(embedding, db, threshold=threshold)

        if match_id:
            print(f"✅ {img_path.name} → {match_id} (score: {score:.3f})")
        else:
            print(f"❌ {img_path.name} → No match (score: {score:.3f})")

if __name__ == "__main__":
    batch_match("images/inputs", "embeddings/criminals.json")
