import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Load the pre-trained TensorFlow model
def load_model():
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    model = hub.load(model_url)
    return model


# Perform object detection on the input image
def detect_objects(image_np, detection_model):
    # Convert the image to unsigned 8-bit integer data type
    image_np = np.uint8(image_np * 255)

    converted_img = tf.convert_to_tensor(image_np)
    converted_img = converted_img[tf.newaxis, ...]
    result = detection_model(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    return result


# Count the number of detected people
def count_people(detections):
    num_detections = int(detections.pop('num_detections'))
    count = 0
    class_ids = detections['detection_classes'][0]
    scores = detections['detection_scores'][0]
    for i in range(num_detections):
        class_id = int(class_ids[i])
        score = scores[i]
        if class_id == 1 and score > 0.5:  # Class ID 1 corresponds to 'person'
            count += 1
    return count



# Draw bounding boxes around detected objects
def draw_boxes(image_np, detections):
    num_detections = int(detections['num_detections'])
    detections = {key: np.squeeze(value) for key, value in detections.items()}

    for i in range(num_detections):
        class_id = int(detections['detection_classes'][i])
        score = detections['detection_scores'][i]
        if class_id == 1 and score > 0.5:  # Class ID 1 corresponds to 'person'
            box = detections['detection_boxes'][i]
            h, w, _ = image_np.shape
            y_min, x_min, y_max, x_max = box
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_np


def main():
    # Load the pre-trained model
    detection_model = load_model()

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide the video file path

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects in the frame
        detections = detect_objects(frame_rgb, detection_model)

        # Draw bounding boxes around detected people
        frame_with_boxes = draw_boxes(frame.copy(), detections)

        # Count the number of detected people
        num_people = count_people(detections)

        # Log the number of detected people in the console
        print("Number of detected people:", num_people)

        # Display the output frame
        cv2.imshow('People Count', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
