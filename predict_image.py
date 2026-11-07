from teachable_machine import TeachableMachine

# Load the trained model
model = TeachableMachine(
    model_path="models/keras_model.h5",
    labels_file_path="models/labels.txt"
)

# Path to the image
image_path = r"data\Koka veda sahasra\18.jpg"

# Predict the person in the image
result = model.classify_image(image_path)

# Display the prediction
print("Predicted person:", result["class_name"])
print("Confidence:", result["class_confidence"])
print("All predictions:", result["predictions"])