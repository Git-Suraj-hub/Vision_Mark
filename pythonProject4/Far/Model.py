import mysql.connector
import numpy as np
import pickle
from sklearn.svm import SVC
import os

# ====================================================
# 1️⃣ Connect to MySQL Database
# ====================================================
conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
cursor = conn.cursor()

# ====================================================
# 2️⃣ Load embeddings grouped by distance category
# ====================================================
print("[INFO] Loading embeddings grouped by distance category...")

cursor.execute("SELECT student_id, distance_category, embedding FROM student_embeddings")
rows = cursor.fetchall()

if not rows:
    print("[ERROR] No embeddings found in database! Please run Embedding.py first.")
    cursor.close()
    conn.close()
    exit()

# Group embeddings by distance category
data_by_distance = {}

for student_id, distance_category, embedding_blob in rows:
    emb = np.frombuffer(embedding_blob, dtype=np.float32)
    if distance_category not in data_by_distance:
        data_by_distance[distance_category] = {"X": [], "y": []}
    data_by_distance[distance_category]["X"].append(emb)
    data_by_distance[distance_category]["y"].append(student_id)

# ====================================================
# 3️⃣ Train a separate SVM for each distance category
# ====================================================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

for distance_category, data in data_by_distance.items():
    X = np.array(data["X"])
    y = np.array(data["y"])

    print(f"\n[INFO] Training SVM for '{distance_category}' with {len(X)} samples...")

    # 🧠 Skip training if only one unique class (not enough to classify)
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"[SKIP] Not enough classes to train SVM for '{distance_category}' "
              f"({len(unique_labels)} class only)")
        continue

    # ✅ Train SVM model
    model = SVC(kernel="rbf", probability=True)
    model.fit(X, y)

    # ✅ Save trained model
    model_path = os.path.join(model_dir, f"svm_{distance_category}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Saved model → {model_path}")

# ====================================================
# 4️⃣ Cleanup
# ====================================================
cursor.close()
conn.close()
print("\n[INFO] ✅ Training complete for all valid distance models!")
