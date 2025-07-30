from src.face_preprocess import detect_and_crop_face
from pathlib import Path

input_dir = Path("images/criminals/")
output_dir = Path("images/cropped_inputs/")

for img_path in input_dir.glob("*.jpg"):
    output_path = output_dir / img_path.name
    print(f"Processing {img_path.name}...")
    face = detect_and_crop_face(str(img_path), str(output_path))
