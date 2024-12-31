import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

# Path constants
GROUP_PHOTO_PATH = "Data/group_photo.jpg"
INDIVIDUAL_PHOTOS_PATH = "Data/people/"
ATTENDANCE_CSV_PATH = "attendance_records/attendance_records.csv"
FACENET_MODEL_PATH = "models/20180402-114759.pb"

# Load FaceNet model
def load_facenet_model():
    model = tf.Graph()
    with model.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(FACENET_MODEL_PATH, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return model

# Extract embeddings using FaceNet
def get_face_embedding(model, face):
    face = cv2.resize(face, (160, 160))
    face = face / 255.0  # Normalize pixel values
    with tf.compat.v1.Session(graph=model) as sess:
        images_placeholder = model.get_tensor_by_name("input:0")
        embeddings = model.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = model.get_tensor_by_name("phase_train:0")
        embedding = sess.run(embeddings, feed_dict={images_placeholder: [face], phase_train_placeholder: False})
    return embedding[0]

# Detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_regions = []
    for face in faces:
        x, y, width, height = face["box"]
        face_regions.append(image[y:y+height, x:x+width])
    return face_regions, faces

# Mark attendance in the CSV file
def mark_attendance(name):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, current_date, current_time])
    print(f"Attendance marked for {name} on {current_date} at {current_time}")

# Load individual photos and create embeddings
def load_individual_embeddings(model):
    known_embeddings = []
    known_names = []
    for file_name in os.listdir(INDIVIDUAL_PHOTOS_PATH):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(INDIVIDUAL_PHOTOS_PATH, file_name)
            image = cv2.imread(image_path)
            faces, _ = detect_faces(image)
            if faces:
                embedding = get_face_embedding(model, faces[0])
                known_embeddings.append(embedding)
                known_names.append(file_name.split(".")[0])  # Use file name as name
    return known_embeddings, known_names

# Match a face embedding with known embeddings
def recognize_face(embedding, known_embeddings, known_names, threshold=1.0):
    distances = [np.linalg.norm(embedding - known_embedding) for known_embedding in known_embeddings]
    min_distance_index = np.argmin(distances)
    if distances[min_distance_index] < threshold:
        return known_names[min_distance_index]
    return "Unknown"

# Visualize attendance data using Matplotlib
def visualize_attendance():
    names, dates, times = [], [], []
    with open(ATTENDANCE_CSV_PATH, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            names.append(row[0])
            dates.append(row[1])
            times.append(row[2])

    plt.figure(figsize=(10, 6))
    plt.scatter(dates, names, c='blue', label="Attendance")
    plt.xlabel("Date")
    plt.ylabel("Name")
    plt.title("Attendance Visualization")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Main function to process the group photo and mark attendance
def process_group_photo():
    facenet_model = load_facenet_model()
    known_embeddings, known_names = load_individual_embeddings(facenet_model)

    # Load the group photo
    group_image = cv2.imread(GROUP_PHOTO_PATH)
    faces, face_boxes = detect_faces(group_image)

    for i, face in enumerate(faces):
        embedding = get_face_embedding(facenet_model, face)
        name = recognize_face(embedding, known_embeddings, known_names)

        # Draw bounding box and label on the group photo
        x, y, width, height = face_boxes[i]["box"]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(group_image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(group_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Mark attendance for recognized faces
        if name != "Unknown":
            mark_attendance(name)

    # Save and display the result
    result_path = "Data/group_photo_result.jpg"
    cv2.imwrite(result_path, group_image)
    print(f"Processed group photo saved at {result_path}")

# Run the attendance system
if __name__ == "__main__":
    process_group_photo()
    visualize_attendance()


