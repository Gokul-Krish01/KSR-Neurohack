import cv2
import numpy as np
import tensorflow as tf

# Load Pre-trained Model (expects images as input)
model = tf.keras.models.load_model("fall_detection_model.h5")  # Update with correct path

# Video Capture
cap = cv2.VideoCapture(0)  # Use webcam

def preprocess_frame(frame):
    """
    Preprocesses the frame for the model: Resize, Normalize, and Reshape.
    """
    image = cv2.resize(frame, (150, 150))  # Resize to model input size
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Reshape to (1, 150, 150, 3)
    return image

def detect_fall(frame):
    """
    Predicts fall using the trained ML model.
    """
    processed_image = preprocess_frame(frame)
    
    # Predict
    prediction = model.predict(processed_image)
    confidence = float(prediction[0][0])  # Extract confidence score

    # Debugging Output
    return ("Fall Detected!", confidence) if confidence > 0.83 else ("No Fall", confidence)

def update_frame():
    """
    Captures video frames and processes fall detection.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural view
        frame = cv2.flip(frame, 1)

        # Detect fall
        status, confidence = detect_fall(frame)

        # Display Status on Frame
        color = (0, 0, 255) if status == "Fall Detected!" else (0, 255, 0)
        cv2.putText(frame, f"{status} ({confidence:.2f})", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Fall Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    update_frame()
