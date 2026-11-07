import cv2
import numpy as np
import pickle

# Load registered face database
with open("models/face_database.pkl", "rb") as file:
    face_database = pickle.load(file)

# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# Load SFace recognizer
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# Recognition threshold (initial value; we'll calibrate it)
THRESHOLD = 0.45

# Start webcam
camera = cv2.VideoCapture(0)

print("Live face recognition started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame not available.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    # Detect all faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:

            # Align and extract face features
            aligned_face = recognizer.alignCrop(frame, face)
            embedding = recognizer.feature(aligned_face)

            best_name = "Unknown"
            best_score = -1

            # Compare against every registered person
            for name, registered_embedding in face_database.items():

                score = recognizer.match(
                    embedding,
                    registered_embedding.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            # Reject weak matches
            if best_score < THRESHOLD:
                best_name = "Unknown"

            x, y, w, h = face[:4].astype(int)

            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Draw name and similarity score
            label = f"{best_name} ({best_score:.2f})"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("AI Friend Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()