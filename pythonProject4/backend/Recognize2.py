import os
import cv2
import numpy as np
import mysql.connector
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
import torch
import time
from scipy.spatial.distance import cosine
from datetime import datetime
from dotenv import load_dotenv
import os
import mysql.connector

# Load .env file
load_dotenv()

class AdvancedFaceRecognition:
    def __init__(self):
        """Initialize all models and database connection"""
        print("[INFO] Initializing Advanced Face Recognition System...")

        # Database connection
   

        # Database connection
        self.conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        self.cursor = self.conn.cursor()
        self.running = False
        self.on_recognize = self.mark_attendance

        # Initialize ArcFace
        print("[INFO] Loading ArcFace model...")
        self.arcface_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.arcface_app.prepare(ctx_id=0, det_size=(640, 640))

        print("[INFO] RetinaFace will be loaded when needed...")

        # Load embeddings with multi-distance support
        self.load_known_faces_multi_distance()

        # Distance thresholds
        self.VERY_CLOSE_THRESHOLD = 150
        self.CLOSE_DISTANCE_THRESHOLD = 80
        self.MID_DISTANCE_THRESHOLD = 40
        self.FAR_DISTANCE_THRESHOLD = 20
        self.VERY_FAR_THRESHOLD = 10

        # Recognition thresholds
        self.RECOGNITION_THRESHOLD_CLOSE = 0.25
        self.RECOGNITION_THRESHOLD_MID = 0.30
        self.RECOGNITION_THRESHOLD_FAR = 0.35
        self.RECOGNITION_THRESHOLD_VERY_FAR = 0.40

        # Performance optimization
        self.last_processed_frame = None
        self.last_faces = []
        self.frame_skip_count = 0

        print("[INFO] System initialized successfully!")

    def load_known_faces_multi_distance(self):
        """Load embeddings from BOTH tables with category-specific fallback"""
        print("[INFO] Loading multi-distance embeddings...")
        
        # PRIMARY: Load individual embeddings from student_embeddings
        self.cursor.execute("""
            SELECT Enrollment_No, distance_category, embedding 
            FROM student_embeddings
            ORDER BY Enrollment_No, distance_category
        """)
        
        self.multi_distance_embeddings = {}
        
        for enrollment_no, category, embedding_blob in self.cursor.fetchall():
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            if enrollment_no not in self.multi_distance_embeddings:
                self.multi_distance_embeddings[enrollment_no] = {}
            
            if category not in self.multi_distance_embeddings[enrollment_no]:
                self.multi_distance_embeddings[enrollment_no][category] = []
            
            self.multi_distance_embeddings[enrollment_no][category].append(embedding)
        
        # FALLBACK: Load category-specific averages from student_detail
        self.cursor.execute("""
            SELECT Name, Enrollment_No, 
                   embedding_very_close, embedding_close, embedding_mid, embedding_far, Embedding
            FROM student_detail
        """)
        
        self.known_faces_category_avg = {}
        
        for row in self.cursor.fetchall():
            name, enrollment_no, emb_vc, emb_c, emb_m, emb_f, emb_global = row
            
            self.known_faces_category_avg[enrollment_no] = {
                'name': name,
                'enrollment_no': enrollment_no
            }
            
            # Load category-specific averages
            if emb_vc:
                emb = np.frombuffer(emb_vc, dtype=np.float32)
                self.known_faces_category_avg[enrollment_no]['very_close'] = emb / np.linalg.norm(emb)
            
            if emb_c:
                emb = np.frombuffer(emb_c, dtype=np.float32)
                self.known_faces_category_avg[enrollment_no]['close'] = emb / np.linalg.norm(emb)
            
            if emb_m:
                emb = np.frombuffer(emb_m, dtype=np.float32)
                self.known_faces_category_avg[enrollment_no]['mid'] = emb / np.linalg.norm(emb)
            
            if emb_f:
                emb = np.frombuffer(emb_f, dtype=np.float32)
                self.known_faces_category_avg[enrollment_no]['far'] = emb / np.linalg.norm(emb)
            
            # Global average (last resort fallback)
            if emb_global:
                emb = np.frombuffer(emb_global, dtype=np.float32)
                self.known_faces_category_avg[enrollment_no]['global'] = emb / np.linalg.norm(emb)
        
        print(f"[INFO] Loaded {len(self.multi_distance_embeddings)} students with individual embeddings")
        print(f"[INFO] Loaded {len(self.known_faces_category_avg)} students with category averages")

    def get_face_distance_category(self, bbox):
        """Determine distance category from bbox"""
        face_width = bbox[2] - bbox[0]
        
        if face_width >= self.VERY_CLOSE_THRESHOLD:
            return "very_close"
        elif face_width >= self.CLOSE_DISTANCE_THRESHOLD:
            return "close"
        elif face_width >= self.MID_DISTANCE_THRESHOLD:
            return "mid"
        elif face_width >= self.FAR_DISTANCE_THRESHOLD:
            return "far"
        else:
            return "very_far"

    def get_adjacent_categories(self, category):
        """Get neighboring distance categories for fallback"""
        category_order = ['very_close', 'close', 'mid', 'far', 'very_far']
        
        try:
            idx = category_order.index(category)
            adjacent = []
            
            if idx > 0:
                adjacent.append(category_order[idx - 1])
            if idx < len(category_order) - 1:
                adjacent.append(category_order[idx + 1])
            
            return adjacent
        except ValueError:
            return []

    def recognize_face_enhanced(self, face_embedding, bbox):
        """
        Multi-level recognition with category-specific fallback
        
        Strategy:
        1. Try individual embeddings from SAME category (student_embeddings)
        2. Try individual embeddings from ADJACENT categories
        3. Try category-specific average from student_detail
        4. Try global average as last resort
        """
        if face_embedding is None:
            return None, 0, "none"
        
        # Normalize input
        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        
        # Determine category
        category = self.get_face_distance_category(bbox)
        face_width = bbox[2] - bbox[0]
        
        best_match = None
        best_similarity = -1
        method_used = "none"
        
        print(f"[DEBUG] Face category: {category} ({face_width:.0f}px)")
        
        # LEVEL 1: Individual embeddings - SAME category
        if self.multi_distance_embeddings:
            for enrollment_no, categories_dict in self.multi_distance_embeddings.items():
                if category in categories_dict:
                    for stored_embedding in categories_dict[category]:
                        similarity = np.dot(face_embedding, stored_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            name = self.known_faces_category_avg.get(enrollment_no, {}).get('name', 'Unknown')
                            best_match = {
                                'name': name,
                                'enrollment_no': enrollment_no
                            }
                            method_used = f"individual_{category}"
        
        # Threshold check for level 1
        threshold = self.get_recognition_threshold(bbox)
        if best_similarity > threshold:
            print(f"[DEBUG] ✅ LEVEL 1: {best_match['name']} ({best_similarity:.3f}) via {method_used}")
            return best_match, best_similarity, method_used
        
        # LEVEL 2: Individual embeddings - ADJACENT categories
        adjacent_categories = self.get_adjacent_categories(category)
        
        for enrollment_no, categories_dict in self.multi_distance_embeddings.items():
            for adj_category in adjacent_categories:
                if adj_category in categories_dict:
                    for stored_embedding in categories_dict[adj_category]:
                        similarity = np.dot(face_embedding, stored_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            name = self.known_faces_category_avg.get(enrollment_no, {}).get('name', 'Unknown')
                            best_match = {
                                'name': name,
                                'enrollment_no': enrollment_no
                            }
                            method_used = f"adjacent_{adj_category}"
        
        # Threshold check for level 2
        if best_similarity > threshold:
            print(f"[DEBUG] ✅ LEVEL 2: {best_match['name']} ({best_similarity:.3f}) via {method_used}")
            return best_match, best_similarity, method_used
        
        # LEVEL 3: Category-specific average
        for enrollment_no, student_data in self.known_faces_category_avg.items():
            # Try exact category average first
            if category in student_data:
                similarity = np.dot(face_embedding, student_data[category])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'name': student_data['name'],
                        'enrollment_no': enrollment_no
                    }
                    method_used = f"category_avg_{category}"
            
            # Try adjacent category averages
            for adj_category in adjacent_categories:
                if adj_category in student_data:
                    similarity = np.dot(face_embedding, student_data[adj_category])
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'name': student_data['name'],
                            'enrollment_no': enrollment_no
                        }
                        method_used = f"category_avg_{adj_category}"
        
        # Threshold check for level 3
        if best_similarity > threshold:
            print(f"[DEBUG] ✅ LEVEL 3: {best_match['name']} ({best_similarity:.3f}) via {method_used}")
            return best_match, best_similarity, method_used
        
        # LEVEL 4: Global average (last resort)
        for enrollment_no, student_data in self.known_faces_category_avg.items():
            if 'global' in student_data:
                similarity = np.dot(face_embedding, student_data['global'])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'name': student_data['name'],
                        'enrollment_no': enrollment_no
                    }
                    method_used = "global_avg"
        
        # Final threshold check
        if best_similarity > threshold:
            print(f"[DEBUG] ✅ LEVEL 4: {best_match['name']} ({best_similarity:.3f}) via {method_used}")
            return best_match, best_similarity, method_used
        
        print(f"[DEBUG] ❌ NO MATCH: Best={best_similarity:.3f}, Threshold={threshold:.3f}")
        return None, best_similarity, method_used

    def recognize_face(self, face_embedding, bbox=None):
        """Wrapper to use enhanced recognition"""
        match, similarity, method = self.recognize_face_enhanced(face_embedding, bbox)
        return match, similarity
    
    def preprocess_face_crop(self, face_crop):
        """Preprocess face crop for better recognition"""
        if face_crop is None or face_crop.size == 0:
            return None
        
        if len(face_crop.shape) == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        elif face_crop.shape[2] == 4:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2RGB)
        
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def detect_faces_very_close(self, img):
        """Very close detection using RetinaFace"""
        try:
            faces = RetinaFace.detect_faces(img, threshold=0.5)
            detected_faces = []

            if isinstance(faces, dict):
                for key, face_data in faces.items():
                    bbox = face_data['facial_area']
                    confidence = face_data['score']

                    face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if face_crop.size > 0:
                        face_crop = self.preprocess_face_crop(face_crop)
                        
                        try:
                            arcface_result = self.arcface_app.get(face_crop)
                            if arcface_result and len(arcface_result) > 0:
                                embedding = arcface_result[0].embedding
                                embedding = embedding / np.linalg.norm(embedding)
                            else:
                                embedding = None
                        except:
                            embedding = None

                        detected_faces.append({
                            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                            'confidence': confidence,
                            'landmarks': face_data.get('landmarks', None),
                            'embedding': embedding,
                            'detection_method': 'RetinaFace (Very Close)'
                        })

            return detected_faces
        except Exception as e:
            print(f"[WARNING] RetinaFace failed: {e}")
            return self.detect_faces_arcface(img, "RetinaFace Failed -> ArcFace")

    def detect_faces_arcface(self, img, method_name="ArcFace"):
        """Standard ArcFace detection"""
        try:
            faces = self.arcface_app.get(img)
            detected_faces = []

            for face in faces:
                bbox = face.bbox.astype(int)
                confidence = face.det_score

                if confidence > 0.2:
                    embedding = face.embedding / np.linalg.norm(face.embedding)
                    
                    detected_faces.append({
                        'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                        'confidence': confidence,
                        'landmarks': face.kps if hasattr(face, 'kps') else None,
                        'embedding': embedding,
                        'detection_method': method_name
                    })

            return detected_faces
        except Exception as e:
            print(f"[WARNING] ArcFace detection failed: {e}")
            return []

    def detect_faces_enhanced(self, img):
        """Enhanced detection for far distances"""
        print("[INFO] Enhancing image for far distance detection...")
        initial_faces = self.detect_faces_arcface(img, "ArcFace (Before Enhancement)")
        return initial_faces

    def get_face_width(self, bbox):
        """Get face width from bbox"""
        return bbox[2] - bbox[0]

    def detect_faces_adaptive(self, img):
        """Adaptive face detection"""
        quick_faces = self.arcface_app.get(img)

        if not quick_faces:
            print("[INFO] No faces detected initially - trying enhancement...")
            return self.detect_faces_enhanced(img)

        face_widths = []
        for face in quick_faces:
            bbox = face.bbox.astype(int)
            face_width = self.get_face_width(bbox)
            face_widths.append(face_width)

        max_face_width = max(face_widths) if face_widths else 0

        print(f"[DEBUG] Max face width: {max_face_width:.1f}px")

        if max_face_width >= self.VERY_CLOSE_THRESHOLD:
            print("[INFO] Using RetinaFace")
            return self.detect_faces_very_close(img)
        elif max_face_width >= self.CLOSE_DISTANCE_THRESHOLD:
            return self.detect_faces_arcface(img, "ArcFace (Close)")
        elif max_face_width >= self.MID_DISTANCE_THRESHOLD:
            return self.detect_faces_arcface(img, "ArcFace (Mid)")
        else:
            return self.detect_faces_enhanced(img)

    def get_recognition_threshold(self, bbox):
        """Get adaptive threshold"""
        distance_category = self.get_face_distance_category(bbox)
        
        threshold_map = {
            'very_close': self.RECOGNITION_THRESHOLD_CLOSE,
            'close': self.RECOGNITION_THRESHOLD_CLOSE,
            'mid': self.RECOGNITION_THRESHOLD_MID,
            'far': self.RECOGNITION_THRESHOLD_FAR,
            'very_far': self.RECOGNITION_THRESHOLD_VERY_FAR
        }
        
        return threshold_map.get(distance_category, self.RECOGNITION_THRESHOLD_MID)

    def draw_results(self, img, faces):
        """Draw detection results"""
        result_img = img.copy()

        for i, face in enumerate(faces):
            bbox = face['bbox']
            confidence = face['confidence']
            method = face.get('detection_method', 'Unknown')

            face_width = self.get_face_width(bbox)
            distance_category = self.get_face_distance_category(bbox)

            color_map = {
                'very_close': (0, 255, 0),
                'close': (0, 255, 255),
                'mid': (255, 165, 0),
                'far': (255, 0, 0),
                'very_far': (128, 0, 128)
            }
            color = color_map.get(distance_category, (255, 255, 255))

            cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            recognition_text = "Unknown"
            recognition_color = (0, 0, 255)
            
            if 'embedding' in face and face['embedding'] is not None:
                match, similarity = self.recognize_face(face['embedding'], bbox)
                if match:
                    recognition_text = f"{match['name']} ({similarity:.2f})"
                    recognition_color = (0, 255, 0)
                else:
                    threshold = self.get_recognition_threshold(bbox)
                    recognition_text = f"Unknown ({similarity:.2f}<{threshold:.2f})"

            label1 = f"{distance_category.upper()} ({face_width:.0f}px)"
            label2 = f"Conf: {confidence:.2f}"
            label3 = recognition_text
            label4 = method

            y_offset = bbox[1] - 70
            cv2.putText(result_img, label1, (bbox[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result_img, label2, (bbox[0], y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result_img, label3, (bbox[0], y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, recognition_color, 2)
            cv2.putText(result_img, label4, (bbox[0], y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if face.get('landmarks') is not None:
                landmarks = face['landmarks']
                if isinstance(landmarks, dict):
                    for point in landmarks.values():
                        cv2.circle(result_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                elif isinstance(landmarks, np.ndarray):
                    for point in landmarks:
                        cv2.circle(result_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        return result_img

    def process_frame(self, img):
        """Process single frame"""
        start_time = time.time()
        faces = self.detect_faces_adaptive(img)
        result_img = self.draw_results(img, faces)

        processing_time = time.time() - start_time
        cv2.putText(result_img, f"Time: {processing_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_img, faces

    def run_webcam(self):
        """Run webcam recognition"""
        print("[INFO] Starting webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.running = True
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 == 0:
                result_frame, faces = self.process_frame(frame)
                self.last_processed_frame = result_frame

                for face in faces:
                    if 'embedding' in face and face['embedding'] is not None:
                        match, similarity = self.recognize_face(face['embedding'], face['bbox'])
                        if match and callable(self.on_recognize):
                            x1, y1, x2, y2 = face['bbox']
                            try:
                                face_crop = frame[y1:y2, x1:x2]
                                _, buffer = cv2.imencode('.jpg', face_crop)
                                img_bytes = buffer.tobytes()
                            except:
                                img_bytes = None

                            try:
                                self.on_recognize(
                                    {'name': match['name'], 'enrollment_no': match['enrollment_no']},
                                    float(similarity), face['bbox'], img_bytes
                                )
                            except Exception as e:
                                print(f"[WARNING] Callback error: {e}")

                display_frame = result_frame
            else:
                display_frame = self.last_processed_frame if self.last_processed_frame is not None else frame

            cv2.imshow('Advanced Face Recognition', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("[INFO] Reloading embeddings...")
                self.load_known_faces_multi_distance()

        self.running = False
        cap.release()
        cv2.destroyAllWindows()

    def stop_webcam(self):
        """Stop webcam"""
        self.running = False
        print("[INFO] Stop signal sent to webcam")

    def process_image(self, image_path):
        """Process single image"""
        print(f"[INFO] Processing: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load: {image_path}")
            return None

        result_img, faces = self.process_frame(img)
        
        # Print analysis
        print(f"\n[INFO] Analysis:")
        print(f"  Total faces: {len(faces)}")
        for i, face in enumerate(faces):
            face_width = self.get_face_width(face['bbox'])
            category = self.get_face_distance_category(face['bbox'])
            print(f"  Face {i+1}: {face_width:.0f}px ({category})")
            
            if 'embedding' in face and face['embedding'] is not None:
                match, similarity = self.recognize_face(face['embedding'], face['bbox'])
                if match:
                    print(f"    ✅ {match['name']} ({similarity:.3f})")
                else:
                    print(f"    ❌ Unknown ({similarity:.3f})")
        
        # Save result
        output_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_img)
        print(f"[INFO] Saved: {output_path}")
        
        cv2.imshow('Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return faces

    def debug_database_embeddings(self):
        """✅ ADDED: Debug database embeddings"""
        print("\n" + "="*60)
        print("DATABASE DEBUGGING")
        print("="*60)
        
        # Check student_embeddings
        self.cursor.execute("""
            SELECT Enrollment_No, distance_category, COUNT(*) as count
            FROM student_embeddings
            GROUP BY Enrollment_No, distance_category
            ORDER BY Enrollment_No, distance_category
        """)
        
        print("\n[student_embeddings] Individual embeddings:")
        current_student = None
        for enrollment_no, category, count in self.cursor.fetchall():
            if enrollment_no != current_student:
                print(f"\n{enrollment_no}:")
                current_student = enrollment_no
            print(f"  - {category:15s}: {count} embeddings")
        
        # Check student_detail
        self.cursor.execute("""
            SELECT Enrollment_No, Name,
                   CASE WHEN embedding_very_close IS NOT NULL THEN '✅' ELSE '❌' END as vc,
                   CASE WHEN embedding_close IS NOT NULL THEN '✅' ELSE '❌' END as c,
                   CASE WHEN embedding_mid IS NOT NULL THEN '✅' ELSE '❌' END as m,
                   CASE WHEN embedding_far IS NOT NULL THEN '✅' ELSE '❌' END as f,
                   CASE WHEN Embedding IS NOT NULL THEN '✅' ELSE '❌' END as g
            FROM student_detail
        """)
        
        print("\n[student_detail] Category averages:")
        print(f"{'Enrollment':<15} | {'Name':<20} | VC | C  | M  | F  | Global")
        print("-"*70)
        for row in self.cursor.fetchall():
            print(f"{row[0]:<15} | {row[1]:<20} | {row[2]}  | {row[3]}  | {row[4]}  | {row[5]}  | {row[6]}")
        
        print("="*60)

    def mark_attendance(self, match_info, similarity, bbox, img_bytes):
        """Mark attendance (callback)"""
        enrollment_no = match_info['enrollment_no']
        
        try:
            cur = self.conn.cursor(dictionary=True)
            now = datetime.now()
            current_time = now.time()
            current_day = now.strftime("%A")

            cur.execute("""
                SELECT Subject, Start_Time, End_Time FROM DummyTimeTable
                WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
            """, (current_day, current_time, current_time))
            
            result = cur.fetchone()

            if not result:
                print("[INFO] No class running")
                cur.close()
                return

            subject = result['Subject']
            start_time = result['Start_Time']
            end_time = result['End_Time']

            subject_table_map = {
                "Theory of Computation": "toc_attendance",
                "Software Engineering": "se_attendence",
                "Database Management System": "dbms_attendence",
                "DBMS LAB": "dbmslab_attendence",
                "SE LAB": "selab_attendance",
                "Personality": "personality_attendance",
                "Introduction to Cyber Security": "ics_attendance",
                "Elective (AI and its Applications)": "ai_mark_attendance",
                "Minor Project": "minorproject_attendance"
            }

            table_name = subject_table_map.get(subject)
            if not table_name:
                cur.close()
                return

            cur.execute(f"""
                SELECT * FROM {table_name}
                WHERE Enrollment_No = %s
                AND DATE(Attendance_DateTime) = CURDATE()
                AND TIME(Attendance_DateTime) BETWEEN %s AND %s
            """, (enrollment_no, start_time, end_time))
            
            if cur.fetchone():
                print(f"[INFO] Already marked: {enrollment_no}")
                cur.close()
                return

            cur.execute(f"""
                INSERT INTO {table_name} (Enrollment_No, Attendance_DateTime, Status)
                VALUES (%s, %s, %s)
            """, (enrollment_no, now, "Present"))
            
            self.conn.commit()
            print(f"[SUCCESS] Marked: {enrollment_no} in {subject}")

        except Exception as e:
            print(f"[ERROR] mark_attendance: {e}")
        finally:
            try:
                cur.close()
            except:
                pass

def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()


# ============================================
# MAIN PROGRAM
# ============================================

def main():
    """Main function with interactive menu"""
    try:
        # Initialize the face recognition system
        face_recognition = AdvancedFaceRecognition()

        print("\n" + "=" * 60)
        print("ADVANCED FACE DETECTION AND RECOGNITION SYSTEM")
        print("=" * 60)
        print("1. Run webcam recognition")
        print("2. Process image file")
        print("3. Process video file")
        print("4. Debug database embeddings")
        print("5. Test recognition with image")
        print("6. Exit")

        while True:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == '1':
                face_recognition.run_webcam()

            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    face_recognition.process_image(image_path)
                else:
                    print("[ERROR] Image file not found!")

            elif choice == '3':
                video_path = input("Enter video path: ").strip()
                if os.path.exists(video_path):
                    print("[INFO] Processing video...")
                    cap = cv2.VideoCapture(video_path)

                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Setup video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        print(f"\rProcessing frame {frame_count}", end="", flush=True)

                        result_frame, _ = face_recognition.process_frame(frame)
                        out.write(result_frame)

                        # Display (optional)
                        cv2.imshow('Video Processing', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    print(f"\n[INFO] Video saved as output_video.mp4")
                else:
                    print("[ERROR] Video file not found!")

            elif choice == '4':
                face_recognition.debug_database_embeddings()

            elif choice == '5':
                image_path = input("Enter test image path: ").strip()
                if os.path.exists(image_path):
                    # Test recognition with detailed output
                    img = cv2.imread(image_path)
                    faces = face_recognition.arcface_app.get(img)

                    if faces:
                        print(f"\n[INFO] Detected {len(faces)} face(s)")
                        for i, face in enumerate(faces):
                            print(f"\nFace {i + 1}:")
                            print(f"  Detection confidence: {face.det_score:.3f}")
                            
                            bbox = face.bbox.astype(int)
                            match, similarity = face_recognition.recognize_face(face.embedding, bbox)
                            threshold = face_recognition.get_recognition_threshold(bbox)
                            
                            if match:
                                print(f"  ✅ RECOGNIZED: {match['name']}")
                                print(f"     Similarity: {similarity:.3f}")
                                print(f"     Threshold: {threshold:.3f}")
                            else:
                                print(f"  ❌ NOT RECOGNIZED")
                                print(f"     Best similarity: {similarity:.3f}")
                                print(f"     Threshold: {threshold:.3f}")
                    else:
                        print("[INFO] No faces detected in image!")
                else:
                    print("[ERROR] Image file not found!")

            elif choice == '6':
                print("\n[INFO] Exiting...")
                break

            else:
                print("[ERROR] Invalid choice! Please enter 1-6.")

    except KeyboardInterrupt:
        print("\n\n[INFO] Process interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()