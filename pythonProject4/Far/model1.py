import mysql.connector
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os

# ============================================
# 1️⃣ Connect to Database
# ============================================
conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
cursor = conn.cursor()
print("[INFO] ✓ Database connected\n")

# ============================================
# 2️⃣ Load Embeddings by Distance
# ============================================
print("="*60)
print("LOADING EMBEDDINGS FROM DATABASE")
print("="*60)

cursor.execute("""
    SELECT student_id, distance_category, embedding 
    FROM student_embeddings
""")
rows = cursor.fetchall()

if not rows:
    print("[ERROR] No embeddings found!")
    exit()

# Group by distance
data_by_distance = {}
for student_id, distance, embedding_blob in rows:
    emb = np.frombuffer(embedding_blob, dtype=np.float32)
    
    if distance not in data_by_distance:
        data_by_distance[distance] = {"X": [], "y": []}
    
    data_by_distance[distance]["X"].append(emb)
    data_by_distance[distance]["y"].append(student_id)

print(f"\n✓ Found {len(data_by_distance)} distance categories")
print(f"✓ Total embeddings: {len(rows)}\n")

# ============================================
# 3️⃣ Train Models
# ============================================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

print("="*60)
print("TRAINING MODELS")
print("="*60)

trained = {}

for distance, data in sorted(data_by_distance.items()):
    X = np.array(data["X"])
    y = np.array(data["y"])
    
    n_samples = len(X)
    n_students = len(np.unique(y))
    
    print(f"\n📍 {distance.upper()}")
    print(f"   Samples: {n_samples}")
    print(f"   Students: {n_students}")
    
    if n_samples < 3 or n_students < 2:
        print(f"   ⚠️  SKIPPED (need min 3 samples, 2 students)")
        continue
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN
    k = min(5, n_samples // 2)
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='cosine',
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Save
    model_file = f"{model_dir}/knn_{distance}.pkl"
    scaler_file = f"{model_dir}/scaler_{distance}.pkl"
    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Saved → knn_{distance}.pkl (k={k})")
    
    trained[distance] = {
        'samples': n_samples,
        'students': n_students,
        'k': k
    }

# ============================================
# 4️⃣ Summary
# ============================================
cursor.close()
conn.close()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nModels trained: {len(trained)}")
for dist, info in sorted(trained.items()):
    print(f"  • {dist}: {info['students']} students, {info['samples']} samples")

print(f"\n💾 Saved in: {model_dir}/")
print("="*60)
print("\n🎯 Next: Run real-time prediction script!")
print("="*60)