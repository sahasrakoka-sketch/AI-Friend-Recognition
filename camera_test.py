import cv2
from teachable_machine import TeachableMachine

# Load your trained model
model = TeachableMachine(
    model_path="models/keras_model.h5",
    labels_file_path="models/labels.txt"
)

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame")
        break

    # Save the current camera frame temporarily
    cv2.imwrite("current_frame.jpg", frame)

    # Predict who is in the frame
    result = model.classify_image("current_frame.jpg")

    name = result["class_name"]
    confidence = result["class_confidence"] * 100

    # Display prediction on camera window
    cv2.putText(
        frame,
        f"{name} ({confidence:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Friend Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()