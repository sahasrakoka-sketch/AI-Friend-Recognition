import cv2
import os
import numpy as np
import pickle

# Load face detection model
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# Load face recognition model
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# Location of friends' photos
DATASET_PATH = "data"

# Store face profiles
face_database = {}

# Process each friend's folder
for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    embeddings = []

    print(f"\nProcessing: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            continue

        # Use the largest detected face
        face = max(faces, key=lambda f: f[2] * f[3])

        aligned_face = recognizer.alignCrop(image, face)
        embedding = recognizer.feature(aligned_face)

        embeddings.append(embedding.flatten())

    if embeddings:
        face_database[person_name] = np.mean(embeddings, axis=0)
        print(f"Registered using {len(embeddings)} images")
    else:
        print("No usable faces found!")

# Save database locally
os.makedirs("models", exist_ok=True)

with open("models/face_database.pkl", "wb") as file:
    pickle.dump(face_database, file)

print("\nFace registration completed!")
print("Registered people:", list(face_database.keys()))