import cv2

# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    height, width = frame.shape[:2]

    # Update detector with actual camera dimensions
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            # Draw a box around each face
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("AI Friend Recognition - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()