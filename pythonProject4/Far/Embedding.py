import os
import cv2
import numpy as np
import mysql.connector
from insightface.app import FaceAnalysis

# ============================================
# STEP 1: CONNECT TO MYSQL DATABASE
# ============================================
# This connects your Python script to MySQL
conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
cursor = conn.cursor()

# ============================================
# STEP 2: INITIALIZE ARCFACE MODEL
# ============================================
# ArcFace (buffalo_l) is used for both face detection & embedding extraction
arcface_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
arcface_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → CPU mode, det_size = image input size


# ============================================
# STEP 3: HELPER FUNCTIONS
# ============================================
veryclose = 0
Closes = 0
Mid = 0
Far = 0

def get_face_distance_category(face_width):
    """
    Determine which distance category a detected face belongs to,
    based on its bounding box width (in pixels).
    Larger faces = person is closer to camera.
    """
    global veryclose, Closes, Mid, Far
    if face_width >= 150:
        veryclose += 1
        return "very_close"
    elif face_width >= 80:
        Closes += 1
        return "close"
    elif face_width >= 40:
        Mid += 1
        return "mid"
    else:
        Far += 1
        return "far"


print(f"Very Close : {veryclose}\n Close : {Closes}\n Mid : {Mid}\n Far : {Far}")
def insert_or_update_embedding(student_id, distance_category, embedding):
    """
    Insert a new embedding or update an existing one in MySQL.
    Each (student_id, distance_category) pair is unique.
    If it already exists → update the embedding.
    """
    embedding_blob = embedding.astype(np.float32).tobytes()  # convert numpy array → bytes for DB storage

    sql = """
        INSERT INTO student_embeddings (student_id, distance_category, embedding)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE embedding=%s
    """
    cursor.execute(sql, (student_id, distance_category, embedding_blob, embedding_blob))
    conn.commit()

    print(f"[INFO] Saved/Updated embedding for {student_id} - {distance_category}")


# ============================================
# STEP 4: DEFINE DATASET FOLDER
# ============================================
# Each student's folder contains their face images
# Example structure:
#   Dataset/
#       MUR2300415_Suraj/
#           1.jpg, 2.jpg, ...
#       MUR2300416_Mahendra/
#           1.jpg, 2.jpg, ...
dataset_folder = r"C:\Users\suraj\Desktop\Project\Advance_Face_Detection_And_Recogination\pythonProject4\Dataset"


# ============================================
# STEP 5: LOOP THROUGH EACH STUDENT FOLDER
# ============================================
for student_folder_name in os.listdir(dataset_folder):
    student_folder_path = os.path.join(dataset_folder, student_folder_name)
    if not os.path.isdir(student_folder_path):
        continue  # skip if it's not a folder

    # Extract student_id (e.g., "MUR2300415" from "MUR2300415_Suraj")
    student_id = student_folder_name.split("_")[0]
    if not student_id:
        continue

    # Dictionary to hold multiple embeddings per distance category
    # Example: {'close': [embed1, embed2, ...], 'mid': [embed3, ...]}
    distance_embeddings = {}

    # ============================================
    # STEP 6: PROCESS EACH IMAGE INSIDE STUDENT FOLDER
    # ============================================
    for image_name in os.listdir(student_folder_path):
        if not image_name.lower().endswith((".jpg", ".png")):
            continue  # skip non-image files

        image_path = os.path.join(student_folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue  # skip unreadable image files

        # Detect faces in the image
        faces = arcface_app.get(img)
        if not faces:
            continue  # skip if no face found

        # ============================================
        # STEP 7: PROCESS EACH DETECTED FACE
        # ============================================
        for face in faces:
            bbox = face.bbox.astype(int)  # bounding box coordinates
            face_width = bbox[2] - bbox[0]  # width of face (used to estimate distance)
            distance_category = get_face_distance_category(face_width)

            # Extract the 512-D embedding and apply L2 normalization
            embedding = face.embedding / np.linalg.norm(face.embedding)

            # Store embeddings grouped by distance type
            if distance_category not in distance_embeddings:
                distance_embeddings[distance_category] = []
            distance_embeddings[distance_category].append(embedding)

    # ============================================
    # STEP 8: COMPUTE AVERAGE EMBEDDING PER DISTANCE
    # ============================================
    # Average helps smooth out noise from multiple images
    for distance_category, embeds in distance_embeddings.items():
        avg_embedding = np.mean(embeds, axis=0)
        insert_or_update_embedding(student_id, distance_category, avg_embedding)


# ============================================
# STEP 9: CLOSE DATABASE CONNECTION
# ============================================
cursor.close()
conn.close()
print("[INFO] All averaged embeddings stored successfully.")
print(f"Very Close : {veryclose}\n Close : {Closes}\n Mid : {Mid}\n Far : {Far}")
