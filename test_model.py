from teachable_machine import TeachableMachine

model = TeachableMachine(
    model_path="models/keras_model.h5",
    labels_file_path="models/labels.txt"
)

print("Model loaded successfully!")
print("Your Teachable Machine model is ready!")