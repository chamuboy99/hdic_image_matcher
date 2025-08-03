import os
from mtcnn import MTCNN
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(filename='face_crop.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

detector = MTCNN()

# Parameters
DATASET_ROOT = 'images/raw'
OUTPUT_ROOT = 'images/cropped'
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
NUM_WORKERS = 8

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS

def align_and_crop_face(image_path, target_size=(160, 160), pad=10):
    img = cv2.imread(image_path)
    if img is None:
        logging.warning(f"⚠️ Failed to read image: {image_path}")
        return None

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_img)

    if not detections:
        logging.info(f"No face detected in {image_path}")
        return None

    largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    keypoints = largest['keypoints']

    try:
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0)
        )

        # Rotate image
        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        # Crop face with padding
        x, y, w, h = largest['box']
        x, y = max(0, x), max(0, y)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(aligned_img.shape[1], x + w + pad)
        y2 = min(aligned_img.shape[0], y + h + pad)

        cropped = aligned_img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, target_size)
        return resized

    except Exception as e:
        logging.warning(f"Alignment failed for {image_path}: {e}")
        return None

def process_image(person_folder, img_file):
    input_path = os.path.join(DATASET_ROOT, person_folder, img_file)
    output_path = os.path.join(OUTPUT_ROOT, person_folder, img_file)

    img = cv2.imread(input_path)
    if img is None:
        logging.warning(f"Failed to read image {input_path}")
        return

    aligned_face = align_and_crop_face(input_path)
    if aligned_face is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, aligned_face)
    else:
        logging.info(f"No aligned face detected in {input_path}")

def main():
    persons = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]

    all_tasks = []
    for person in persons:
        img_files = [f for f in os.listdir(os.path.join(DATASET_ROOT, person)) if is_image_file(f)]
        for img_file in img_files:
            all_tasks.append((person, img_file))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_image, person, img_file): (person, img_file) for person, img_file in all_tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                future.result()
            except Exception as e:
                person, img_file = futures[future]
                logging.error(f"Error processing {person}/{img_file}: {e}")

    print("Face cropping and alignment completed.")

if __name__ == '__main__':
    main()
