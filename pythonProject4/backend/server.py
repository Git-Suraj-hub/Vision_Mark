# backend/server.py
import os
import io
import re
import json
import time
import secrets
import datetime
import threading
import base64
from typing import Optional, List
from fastapi import FastAPI, Request, Form, Cookie, Response, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mysql.connector
from contextlib import asynccontextmanager
import cv2
import numpy as np
from dotenv import load_dotenv
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Load .env file
load_dotenv()
# Import recognition class
from backend.Recognize2 import AdvancedFaceRecognition

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
CAPTURE_DIR = os.path.join(STATIC_DIR, "captures")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

# Create directories
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Session management
sessions = {}
SESSION_EXPIRE_SECONDS = 60 * 60 * 6  # 6 hours

def create_session(userinfo: dict):
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "userinfo": userinfo, 
        "expires": datetime.datetime.utcnow() + datetime.timedelta(seconds=SESSION_EXPIRE_SECONDS)
    }
    return token

def get_session(token: Optional[str]):
    if not token:
        return None
    s = sessions.get(token)
    if not s:
        return None
    if s["expires"] < datetime.datetime.utcnow():
        del sessions[token]
        return None
    return s["userinfo"]

# Database helper (Supports local MySQL & TiDB Cloud)
def get_db_conn():
    port = int(os.getenv("DB_PORT", 4000 if "tidb" in (os.getenv("DB_HOST") or "").lower() else 3306))
    ssl_disabled = (os.getenv("DB_SSL_DISABLED") or "False").lower() in ("true", "1")
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"),
        port=port,
        user=os.getenv("DB_USER", "3QtZDLSqiksoXFK.root"),
        password=os.getenv("DB_PASSWORD", "Co5onzrOD2uxx23U"),
        database=os.getenv("DB_NAME", "face_detection"),
        ssl_disabled=ssl_disabled
    )

# Subject to table mapping (B.Tech CSE-7th Sem Mewar University)
SUBJECT_TABLE_MAP = {
    "Fuzzy Set Theory and Its Application": "fuzzy_attendance",
    "Fuzzy Set Theory and its Application": "fuzzy_attendance",
    "Fuzzy Set Theory": "fuzzy_attendance",
    "Soft Skills and Interpersonal Communications": "softskills_attendance",
    "Soft Skills and Enter personal Communications": "softskills_attendance",
    "Soft Skills": "softskills_attendance",
    "Remote Sensing and GIS": "remotesensing_attendance",
    "Remote Sensing and G S": "remotesensing_attendance",
    "Remote Sensing": "remotesensing_attendance",
    "Computer Network Security and Cryptography": "cryptography_attendance",
    "Network Security and Cryptography": "cryptography_attendance",
    "CNSC": "cryptography_attendance",
    "Data Analysis": "dataanalysis_attendance",
    "Major Project": "majorproject_attendance",
    "Major Project / Seminar": "majorproject_attendance"
}

CANONICAL_SUBJECTS = [
    ("Fuzzy Set Theory and Its Application", "fuzzy_attendance"),
    ("Soft Skills and Interpersonal Communications", "softskills_attendance"),
    ("Remote Sensing and GIS", "remotesensing_attendance"),
    ("Computer Network Security and Cryptography", "cryptography_attendance"),
    ("Data Analysis", "dataanalysis_attendance"),
    ("Major Project", "majorproject_attendance")
]

# Global recognition object
recog = None
recog_thread = None
current_recognitions = []  # Store recent recognitions
active_camera_teacher = None  # Dict of teacher userinfo who started the camera

# Recognition callback with Timetable Enforcement & Testing Mode
def recognition_callback(match, similarity, bbox, img_bytes):
    """
    Called when face is recognized during webcam stream.
    Enforces university timetable rules:
    - If current timetable slot matches the teacher's subject: Records official attendance!
    - If current timetable slot does NOT match (different class or free hour): Runs in PREVIEW/TESTING MODE only (does NOT record attendance).
    """
    try:
        enrollment_no = match.get("enrollment_no")
        name = match.get("name")
        
        now = datetime.datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        exact_time_str = now.strftime("%I:%M:%S %p")
        
        # Save captured image
        image_path = None
        if img_bytes:
            fname = f"{enrollment_no}_{int(now.timestamp())}.jpg"
            image_path = os.path.join("captures", fname)
            full_path = os.path.join(CAPTURE_DIR, fname)
            with open(full_path, "wb") as f:
                f.write(img_bytes)
                
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Query currently active class from timetable
        cur.execute("""
            SELECT Subject, Start_Time, End_Time, Teacher, Room FROM timetable
            WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
        """, (current_day, current_time, current_time))
        current_class = cur.fetchone()
        
        # Check active camera teacher who is running the webcam
        teacher_sub = active_camera_teacher.get("subject") if active_camera_teacher else None
        teacher_name = active_camera_teacher.get("name") or (active_camera_teacher.get("username") if active_camera_teacher else "Faculty")
        teacher_table = SUBJECT_TABLE_MAP.get(teacher_sub) if teacher_sub else None
        
        is_live_attendance = False
        mode_note = "Testing Preview Mode"
        
        if current_class:
            running_subject = current_class['Subject']
            running_teacher = current_class.get('Teacher') or 'Faculty'
            curr_table = SUBJECT_TABLE_MAP.get(running_subject)
            
            if teacher_table and curr_table and teacher_table == curr_table:
                is_live_attendance = True
                mode_note = f"Recorded for {running_subject}"
            else:
                mode_note = f"Testing Mode: Active class is '{running_subject}' [{running_teacher}], not '{teacher_sub}'"
        else:
            mode_note = f"Testing Mode: No lecture scheduled right now ({current_day} {exact_time_str})"
            
        if not is_live_attendance:
            # PREVIEW / TESTING MODE ONLY - DO NOT MARK ATTENDANCE
            current_recognitions.append({
                'enrollment_no': enrollment_no,
                'name': name,
                'subject': teacher_sub or "Preview",
                'time': exact_time_str,
                'confidence': float(similarity),
                'image_path': image_path,
                'status': 'Testing Preview (Not Marked)',
                'mode': 'testing',
                'note': mode_note
            })
            if len(current_recognitions) > 50:
                current_recognitions.pop(0)
            print(f"[TESTING PREVIEW] {name} ({enrollment_no}) detected at {exact_time_str}. {mode_note}. Attendance NOT marked.")
            cur.close()
            conn.close()
            return
            
        # LIVE ATTENDANCE MODE - Class matches teacher's subject!
        start_time = current_class['Start_Time']
        end_time = current_class['End_Time']
        table_name = teacher_table
        
        cur.execute(f"""
            SELECT * FROM {table_name}
            WHERE Enrollment_No = %s
            AND DATE(Attendance_DateTime) = CURDATE()
            AND TIME(Attendance_DateTime) BETWEEN %s AND %s
        """, (enrollment_no, start_time, end_time))
        
        already_marked = cur.fetchone()
        
        if already_marked:
            print(f"[INFO] {enrollment_no} already marked for {teacher_sub} in slot {start_time}-{end_time}")
            cur.close()
            conn.close()
            return
            
        # Mark official live attendance
        official_remarks = f"AI Facial Biometric Verification (ArcFace {int(float(similarity)*100)}%)"
        cur.execute(f"""
            INSERT INTO {table_name} 
            (Enrollment_No, Attendance_DateTime, Status, recognized_by, confidence, image_path, remarks)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (enrollment_no, now, "Present", "AI Biometric (ArcFace)", float(similarity), image_path, official_remarks))
        conn.commit()
        
        current_recognitions.append({
            'enrollment_no': enrollment_no,
            'name': name,
            'subject': teacher_sub,
            'time': exact_time_str,
            'confidence': float(similarity),
            'image_path': image_path,
            'status': 'Present (Recorded)',
            'mode': 'live',
            'note': 'Marked Present in Scheduled Lecture'
        })
        if len(current_recognitions) > 50:
            current_recognitions.pop(0)
            
        print(f"[LIVE ATTENDANCE] Successfully marked attendance for {name} ({enrollment_no}) in {teacher_sub} at {exact_time_str}")
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"[ERROR] recognition_callback: {e}")
        import traceback
        traceback.print_exc()

# Automated Student Face Dataset & Embedding Generator
def register_student_face_dataset(enrollment_no: str, student_name: str, saved_image_paths: list[str]) -> dict:
    """
    Automated pipeline:
    1. Reads saved images from Dataset/{enrollment_no}_{name}/
    2. Runs ArcFace detection & embedding extraction
    3. Categorizes each face by distance
    4. Inserts into student_embeddings
    5. Calculates category-specific and global averages
    6. Updates student_detail table
    7. Reloads live recognition cache
    """
    global recog
    arcface = None
    if recog and hasattr(recog, 'arcface_app') and recog.arcface_app is not None:
        arcface = recog.arcface_app
    else:
        from insightface.app import FaceAnalysis
        arcface = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        arcface.prepare(ctx_id=0, det_size=(640, 640))
        
    DISTANCE_THRESHOLDS = {
        'very_close': 150,
        'close': 80,
        'mid': 40,
        'far': 20
    }
    
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    
    category_counts = {'very_close': 0, 'close': 0, 'mid': 0, 'far': 0, 'very_far': 0}
    total_faces = 0
    
    for img_idx, img_path in enumerate(saved_image_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        try:
            faces = arcface.get(img)
        except Exception as e:
            print(f"[ERROR] ArcFace error on {img_path}: {e}")
            continue
            
        if not faces:
            continue
            
        for f_idx, face in enumerate(faces):
            total_faces += 1
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            
            if face_width >= DISTANCE_THRESHOLDS['very_close']:
                cat = 'very_close'
            elif face_width >= DISTANCE_THRESHOLDS['close']:
                cat = 'close'
            elif face_width >= DISTANCE_THRESHOLDS['mid']:
                cat = 'mid'
            elif face_width >= DISTANCE_THRESHOLDS['far']:
                cat = 'far'
            else:
                cat = 'very_far'
                
            emb = face.embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            emb_blob = emb.tobytes()
            
            img_source = f"photo_{img_idx}_face_{f_idx}"
            cur.execute("""
                INSERT INTO student_embeddings 
                (Enrollment_No, distance_category, embedding, image_source)
                VALUES (%s, %s, %s, %s)
            """, (enrollment_no, cat, emb_blob, img_source))
            category_counts[cat] += 1
            
    conn.commit()
    
    # Calculate category averages and global average
    categories_to_calc = ['very_close', 'close', 'mid', 'far']
    category_averages = {}
    
    for cat in categories_to_calc:
        cur.execute("""
            SELECT embedding FROM student_embeddings
            WHERE Enrollment_No = %s AND distance_category = %s
        """, (enrollment_no, cat))
        rows = cur.fetchall()
        if rows:
            embs = []
            for r in rows:
                v = np.frombuffer(r['embedding'], dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    embs.append(v / norm)
            if embs:
                cat_avg = np.mean(embs, axis=0)
                cat_avg = cat_avg / np.linalg.norm(cat_avg)
                category_averages[cat] = cat_avg
                
    if category_averages:
        global_avg = np.mean(list(category_averages.values()), axis=0)
        global_avg = global_avg / np.linalg.norm(global_avg)
    else:
        cur.execute("SELECT embedding FROM student_embeddings WHERE Enrollment_No = %s", (enrollment_no,))
        rows = cur.fetchall()
        if rows:
            embs = [np.frombuffer(r['embedding'], dtype=np.float32) for r in rows]
            global_avg = np.mean(embs, axis=0)
            global_avg = global_avg / np.linalg.norm(global_avg)
        else:
            global_avg = np.zeros(512, dtype=np.float32)
            
    # Update student_detail with averages
    cur.execute("""
        UPDATE student_detail 
        SET Name = %s,
            Embedding = %s,
            embedding_very_close = %s,
            embedding_close = %s,
            embedding_mid = %s,
            embedding_far = %s
        WHERE Enrollment_No = %s
    """, (
        student_name,
        global_avg.astype(np.float32).tobytes(),
        category_averages.get('very_close', np.zeros(512, dtype=np.float32)).astype(np.float32).tobytes(),
        category_averages.get('close', np.zeros(512, dtype=np.float32)).astype(np.float32).tobytes(),
        category_averages.get('mid', np.zeros(512, dtype=np.float32)).astype(np.float32).tobytes(),
        category_averages.get('far', np.zeros(512, dtype=np.float32)).astype(np.float32).tobytes(),
        enrollment_no
    ))
    conn.commit()
    cur.close()
    conn.close()
    
    if recog and hasattr(recog, 'load_known_faces_multi_distance'):
        try:
            recog.load_known_faces_multi_distance()
            print(f"[INFO] Reloaded live recognition cache for {enrollment_no}")
        except Exception as e:
            print(f"[WARN] Error reloading recognition cache: {e}")
            
    return {
        "total_faces": total_faces,
        "category_counts": category_counts
    }

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recog
    print("[INFO] Starting up face recognition system...")
    recog = AdvancedFaceRecognition()
    recog.on_recognize = recognition_callback
    print("[INFO] Face recognition system loaded successfully!")
    yield
    print("[INFO] Shutting down system...")
    if recog:
        recog.stop_webcam()
    recog = None

app = FastAPI(lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login/student", response_class=HTMLResponse)
def login_student_page(request: Request):
    return templates.TemplateResponse("login_student.html", {"request": request})

@app.get("/login/teacher", response_class=HTMLResponse)
def login_teacher_page(request: Request):
    return templates.TemplateResponse("login_teacher.html", {"request": request})

@app.get("/student/dashboard", response_class=HTMLResponse)
def student_dashboard(request: Request, session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "student":
        return RedirectResponse("/login/student")
    return templates.TemplateResponse("student.html", {"request": request, "student": user})

@app.get("/teacher/dashboard", response_class=HTMLResponse)
def teacher_dashboard(request: Request, session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        return RedirectResponse("/login/teacher")
    return templates.TemplateResponse("teacher.html", {"request": request, "teacher": user})

# ==================== API ENDPOINTS ====================

@app.post("/api/login")
def api_login(role: str = Form(...), username: str = Form(...), password: str = Form(...)):
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    
    if role == "student":
        cur.execute("SELECT * FROM students WHERE username=%s", (username,))
    else:
        cur.execute("SELECT * FROM teachers WHERE username=%s", (username,))
    
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        return JSONResponse({"success": False, "msg": "User not found"})
    
    if row.get("password") != password:
        return JSONResponse({"success": False, "msg": "Invalid credentials"})
    
    userinfo = {
        "role": role,
        "username": username,
        "name": row.get("name"),
        "enrollment_no": row.get("enrollment_no") if role == "student" else None,
        "class": row.get("class"),
        "subject": row.get("subject") if role == "teacher" else None
    }
    
    token = create_session(userinfo)
    resp = JSONResponse({"success": True})
    resp.set_cookie(key="session", value=token, httponly=True, samesite="lax")
    return resp

@app.post("/api/logout")
def api_logout(session: Optional[str] = Cookie(None)):
    if session and session in sessions:
        del sessions[session]
    resp = JSONResponse({"success": True})
    resp.delete_cookie("session")
    return resp

@app.get("/api/current_user")
def api_current_user(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    return {"user": user}

@app.get("/api/current_subject")
def api_current_subject():
    """Get currently running subject from timetable"""
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        now = datetime.datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        cur.execute("""
            SELECT Subject, Start_Time, End_Time, Teacher, Room 
            FROM timetable
            WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
        """, (current_day, current_time, current_time))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return {
                "success": True,
                "subject": result['Subject'],
                "start_time": str(result['Start_Time']),
                "end_time": str(result['End_Time']),
                "teacher": result.get('Teacher'),
                "room": result.get('Room')
            }
        else:
            return {"success": False, "msg": "No class running"}
            
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/teacher/class_status")
def api_teacher_class_status(session: Optional[str] = Cookie(None)):
    """Check whether teacher's assigned subject is currently scheduled in timetable"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    teacher_name = user.get("name", user.get("username"))
    teacher_table = SUBJECT_TABLE_MAP.get(teacher_subject)
    
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    
    now = datetime.datetime.now()
    current_time = now.time()
    current_day = now.strftime("%A")
    exact_time_display = now.strftime("%I:%M %p")
    
    cur.execute("""
        SELECT Subject, Start_Time, End_Time, Teacher, Room 
        FROM timetable
        WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
    """, (current_day, current_time, current_time))
    
    current_class = cur.fetchone()
    cur.close()
    conn.close()
    
    if current_class:
        running_subject = current_class['Subject']
        running_teacher = current_class.get('Teacher') or 'Faculty'
        room = current_class.get('Room') or "MB-119"
        start_t = str(current_class['Start_Time'])
        end_t = str(current_class['End_Time'])
        
        curr_table = SUBJECT_TABLE_MAP.get(running_subject)
        
        if curr_table and teacher_table and curr_table == teacher_table:
            # Active class matches teacher's subject!
            return {
                "success": True,
                "is_scheduled_now": True,
                "mode": "LIVE_ATTENDANCE",
                "badge": f"● LIVE ATTENDANCE ACTIVE — {teacher_subject}",
                "title": f"🟢 Active Lecture: {running_subject} ({start_t} - {end_t}, Room {room})",
                "message": f"Aapka assigned subject '{teacher_subject}' actively schedule me chal raha hai. Camera biometric face recognition dwara attendance automatically mark kar raha hai!",
                "current_class": {
                    "subject": running_subject,
                    "teacher": running_teacher,
                    "room": room,
                    "start_time": start_t,
                    "end_time": end_t
                },
                "teacher_subject": teacher_subject
            }
        else:
            # Different class is running!
            return {
                "success": True,
                "is_scheduled_now": False,
                "mode": "TESTING_PREVIEW_MODE",
                "badge": f"⚠️ PREVIEW ONLY (NOT YOUR LECTURE: {teacher_subject})",
                "title": f"⚠️ Scheduled Subject Mismatch: Active lecture is '{running_subject}'",
                "message": f"Aapka assigned subject '{teacher_subject}' hai, lekin abhi timetable me '{running_subject}' (Faculty: {running_teacher}, Room {room}, {start_t} - {end_t}) chal raha hai. Aapka webcam Testing/Preview Mode me chalega — is samay '{teacher_subject}' ki attendance auto-mark nahi hogi. Kisi student ki attendance lagane ke liye neeche 'Manual Faculty Attendance' use karein.",
                "current_class": {
                    "subject": running_subject,
                    "teacher": running_teacher,
                    "room": room,
                    "start_time": start_t,
                    "end_time": end_t
                },
                "teacher_subject": teacher_subject
            }
    else:
        # No class running now
        return {
            "success": True,
            "is_scheduled_now": False,
            "mode": "TESTING_PREVIEW_MODE",
            "badge": f"⚠️ FREE PERIOD / PREVIEW — {teacher_subject}",
            "title": f"⚠️ Free Period / No Active Class — Assigned Subject: '{teacher_subject}'",
            "message": f"Abhi ({current_day}, {exact_time_display}) timetable me koi university lecture scheduled nahi hai. Aapka assigned subject '{teacher_subject}' hai. Camera sirf Testing / Preview Mode me chalega (attendance database me save nahi hogi). Unscheduled lecture ki attendance ke liye neeche 'Manual Faculty Attendance' use karein.",
            "current_class": None,
            "teacher_subject": teacher_subject
        }

@app.get("/api/student/attendance/{enrollment_no}")
def api_student_attendance(enrollment_no: str):
    """Get student attendance across all canonical subjects without duplicate aliases"""
    enrollment_no = enrollment_no.strip().upper()
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        attendance_data = {}
        
        for subject_name, table_name in CANONICAL_SUBJECTS:
            try:
                cur.execute(f"""
                    SELECT id, Enrollment_No, Attendance_DateTime, Status, confidence, image_path, recognized_by, remarks
                    FROM {table_name}
                    WHERE Enrollment_No = %s
                    ORDER BY Attendance_DateTime DESC
                    LIMIT 200
                """, (enrollment_no,))
                
                records = cur.fetchall()
                attendance_data[subject_name] = [
                    {
                        'date': r['Attendance_DateTime'].strftime("%Y-%m-%d"),
                        'display_date': r['Attendance_DateTime'].strftime("%d-%b-%Y"),
                        'time': r['Attendance_DateTime'].strftime("%I:%M:%S %p"),
                        'raw_time': r['Attendance_DateTime'].strftime("%H:%M:%S"),
                        'day': r['Attendance_DateTime'].strftime("%A"),
                        'status': r['Status'],
                        'recognized_by': r.get('recognized_by') or 'AI Face Biometrics',
                        'confidence': float(r.get('confidence') if r.get('confidence') is not None else 1.0),
                        'remarks': r.get('remarks') or ('AI Biometric Verification' if 'ai' in str(r.get('recognized_by') or '').lower() else 'Manual Entry'),
                        'image_path': r.get('image_path')
                    } for r in records
                ]
            except Exception as e:
                print(f"[ERROR] Getting attendance from {table_name}: {e}")
                attendance_data[subject_name] = []
        
        cur.close()
        conn.close()
        
        return {"success": True, "attendance": attendance_data}
        
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/teacher/attendance")
def api_teacher_attendance(session: Optional[str] = Cookie(None)):
    """Get attendance for teacher's subject only with accurate student names and formatted timestamps"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    if not teacher_subject:
        return {"success": False, "msg": "No subject assigned to teacher"}
    
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        return {"success": False, "msg": "Invalid subject"}
    
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Get all attendance for this subject joined with student_detail & students for clean names
        cur.execute(f"""
            SELECT a.*, COALESCE(s.Name, st.name, a.Enrollment_No) as student_name
            FROM {table_name} a
            LEFT JOIN student_detail s ON a.Enrollment_No = s.Enrollment_No
            LEFT JOIN students st ON a.Enrollment_No = st.enrollment_no
            ORDER BY a.Attendance_DateTime DESC
            LIMIT 500
        """)
        
        records = cur.fetchall()
        
        # Get statistics
        cur.execute(f"""
            SELECT 
                COUNT(DISTINCT Enrollment_No) as total_students,
                COUNT(*) as total_attendance,
                COUNT(DISTINCT DATE(Attendance_DateTime)) as total_days
            FROM {table_name}
        """)
        
        stats = cur.fetchone()
        
        cur.close()
        conn.close()
        
        formatted_records = [
            {
                'id': r['id'],
                'enrollment_no': r['Enrollment_No'],
                'name': r.get('student_name', 'Student'),
                'date': r['Attendance_DateTime'].strftime("%d-%b-%Y"),
                'raw_date': r['Attendance_DateTime'].strftime("%Y-%m-%d"),
                'time': r['Attendance_DateTime'].strftime("%I:%M:%S %p"),
                'day': r['Attendance_DateTime'].strftime("%A"),
                'status': r['Status'],
                'recognized_by': r.get('recognized_by') or 'Manual',
                'confidence': float(r.get('confidence') if r.get('confidence') is not None else 1.0),
                'remarks': r.get('remarks') or ('AI Biometric Verification' if 'ai' in str(r.get('recognized_by') or '').lower() else 'Manual Entry'),
                'image_path': r.get('image_path')
            } for r in records
        ]
        
        return {
            "success": True,
            "subject": teacher_subject,
            "records": formatted_records,
            "statistics": stats
        }
        
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.post("/api/teacher/delete_attendance")
def api_teacher_delete_attendance(
    record_id: int = Form(...),
    session: Optional[str] = Cookie(None)
):
    """Delete a specific attendance record by ID for the teacher's course"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        raise HTTPException(status_code=400, detail="Invalid subject")
        
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(f"SELECT id, Enrollment_No, Attendance_DateTime, Status FROM {table_name} WHERE id = %s", (record_id,))
        rec = cur.fetchone()
        if not rec:
            cur.close()
            conn.close()
            return {"success": False, "msg": "Attendance record not found"}
            
        cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (record_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        enroll = rec['Enrollment_No']
        dt_str = rec['Attendance_DateTime'].strftime("%d-%b-%Y %I:%M %p")
        return {
            "success": True, 
            "msg": f"✅ Attendance record for {enroll} ({dt_str}) removed successfully!"
        }
    except Exception as e:
        return {"success": False, "msg": f"❌ Error deleting record: {str(e)}"}

@app.post("/api/mark_attendance")
def api_mark_attendance(
    enrollment_no: str = Form(...),
    status: str = Form("Present"),
    remark_category: Optional[str] = Form("Direct Faculty Adjustment"),
    custom_remark: Optional[str] = Form(None),
    att_date: Optional[str] = Form(None),
    att_time: Optional[str] = Form(None),
    override_timetable: bool = Form(False),
    session: Optional[str] = Cookie(None)
):
    """
    Manual attendance marking with timetable validation or direct teacher override,
    recording exact submission timestamp, custom lecture date/time, and official faculty remarks/reasons.
    """
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    teacher_name = user.get("name", user.get("username"))
    
    if not teacher_subject:
        return {"success": False, "msg": "No subject assigned to you"}
    
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        return {"success": False, "msg": f"Invalid subject or table mapping: {teacher_subject}"}

    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        now = datetime.datetime.now()
        if att_date and att_date.strip():
            date_str = att_date.strip()
            time_str = att_time.strip() if (att_time and att_time.strip()) else now.strftime("%H:%M:%S")
            if len(time_str) == 5:
                time_str += ":00"
            try:
                now = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        current_time = now.time()
        current_day = now.strftime("%A")
        exact_time_display = now.strftime("%I:%M:%S %p")
        display_date = now.strftime("%d-%b-%Y")
        
        start_time = None
        end_time = None
        
        if not override_timetable:
            # ============================================
            # STEP 1: Check if ANY class is running now
            # ============================================
            cur.execute("""
                SELECT Subject, Start_Time, End_Time, Teacher, Room 
                FROM timetable
                WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
            """, (current_day, current_time, current_time))
            
            current_class = cur.fetchone()
            
            if not current_class:
                cur.close()
                conn.close()
                return {
                    "success": False, 
                    "msg": f"❌ Class Not Scheduled: Is samay aapki class nahi hai!\n\n"
                           f"According to the official timetable, no lecture is scheduled on {current_day} at {exact_time_display} (Free Period / Off-Hours).\n\n"
                           f"Please check the university timetable and select your scheduled lecture slot."
                }
            
            current_subject = current_class['Subject']
            start_time = current_class['Start_Time']
            end_time = current_class['End_Time']
            
            # ============================================
            # STEP 2: Check if teacher's subject matches current class
            # ============================================
            mapped_curr_table = SUBJECT_TABLE_MAP.get(current_subject)
            if mapped_curr_table != table_name and teacher_subject != current_subject:
                cur.close()
                conn.close()
                return {
                    "success": False,
                    "msg": f"❌ Timetable Conflict: Is samay aapki class nahi hai!\n\n"
                           f"Selected Slot: {current_day} at {exact_time_display}\n"
                           f"Scheduled Class: {current_subject} (Teacher: {current_class['Teacher']}, Room: {current_class['Room']})\n"
                           f"Your Subject: {teacher_subject}\n\n"
                           f"Attendance for '{teacher_subject}' cannot be marked right now. Please check the timetable and mark during your subject's scheduled slot.",
                    "current_subject": current_subject,
                    "teacher_subject": teacher_subject
                }
        
        # ============================================
        # STEP 3: Check if student exists
        # ============================================
        cur.execute("""
            SELECT Name FROM student_detail WHERE Enrollment_No = %s
        """, (enrollment_no,))
        
        student = cur.fetchone()
        if not student:
            cur.execute("SELECT name FROM students WHERE enrollment_no = %s", (enrollment_no,))
            student_fallback = cur.fetchone()
            if student_fallback:
                student_name = student_fallback['name']
                cur.execute("INSERT INTO student_detail (Enrollment_No, Name) VALUES (%s, %s)", (enrollment_no, student_name))
                conn.commit()
            else:
                cur.close()
                conn.close()
                return {
                    "success": False,
                    "msg": f"❌ Student {enrollment_no} not found in database. Please register the student first."
                }
        else:
            student_name = student['Name']
        
        # ============================================
        # STEP 4: Check for duplicate
        # ============================================
        if not override_timetable and start_time and end_time:
            cur.execute(f"""
                SELECT * FROM {table_name}
                WHERE Enrollment_No = %s
                AND DATE(Attendance_DateTime) = DATE(%s)
                AND TIME(Attendance_DateTime) BETWEEN %s AND %s
            """, (enrollment_no, now, start_time, end_time))
        else:
            # Direct mode: check within 20 minutes of target datetime
            cur.execute(f"""
                SELECT * FROM {table_name}
                WHERE Enrollment_No = %s
                AND DATE(Attendance_DateTime) = DATE(%s)
                AND TIMESTAMPDIFF(MINUTE, Attendance_DateTime, %s) BETWEEN -20 AND 20
            """, (enrollment_no, now, now))
        
        existing = cur.fetchone()
        
        if existing:
            existing_time = existing['Attendance_DateTime'].strftime("%I:%M:%S %p")
            existing_method = existing.get('recognized_by', 'unknown')
            
            cur.close()
            conn.close()
            
            slot_info = f"({start_time} - {end_time})" if (start_time and end_time) else "(Recent 20 mins)"
            return {
                "success": False,
                "msg": f"⚠️ Attendance already marked!\n\n"
                       f"Student: {student_name} ({enrollment_no})\n"
                       f"Subject: {teacher_subject}\n"
                       f"Time: {existing_time}\n"
                       f"Method: {existing_method}\n\n"
                       f"Cannot mark duplicate attendance in the same period {slot_info}.",
                "already_marked": True,
                "existing_record": {
                    "time": existing_time,
                    "method": existing_method
                }
            }
        
        # ============================================
        # STEP 5: Prepare remarks & insert attendance record
        # ============================================
        remark_parts = []
        if remark_category and remark_category.strip():
            remark_parts.append(remark_category.strip())
        if custom_remark and custom_remark.strip():
            remark_parts.append(custom_remark.strip())
        final_remarks = " — ".join(remark_parts) if remark_parts else "Manual Faculty Override"
        
        method_str = f"Manual Entry ({teacher_name})"
        
        cur.execute(f"""
            INSERT INTO {table_name}
            (Enrollment_No, Attendance_DateTime, Status, recognized_by, confidence, remarks)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (enrollment_no, now, status, method_str, 1.0, final_remarks))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "msg": f"✅ Attendance marked successfully!\n\n"
                   f"Student: {student_name} ({enrollment_no})\n"
                   f"Subject: {teacher_subject}\n"
                   f"Status: {status}\n"
                   f"Time: {exact_time_display}\n"
                   f"Method: {method_str}\n"
                   f"Reason / Remarks: {final_remarks}",
            "record": {
                "student_name": student_name,
                "enrollment_no": enrollment_no,
                "subject": teacher_subject,
                "status": status,
                "time": exact_time_display,
                "method": method_str,
                "remarks": final_remarks
            }
        }
        
    except Exception as e:
        return {
            "success": False, 
            "msg": f"❌ Database error: {str(e)}"
        }

# ==================== TEACHER ANALYTICS & REPORTS ====================

@app.get("/api/teacher/analytics")
def api_teacher_analytics(
    period: str = "all",
    session: Optional[str] = Cookie(None)
):
    """Get rich real-world analytics with daily, weekly, monthly filters and day-of-week bunk trends"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        return {"success": False, "msg": "Invalid subject"}
    
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        where_clause = ""
        if period == "daily":
            where_clause = "WHERE DATE(Attendance_DateTime) = CURDATE()"
        elif period == "weekly":
            where_clause = "WHERE Attendance_DateTime >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
        elif period == "monthly":
            where_clause = "WHERE Attendance_DateTime >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
            
        # 1. Total distinct sessions/dates for this period
        cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} {where_clause} ORDER BY att_date ASC")
        all_dates = [r['att_date'] for r in cur.fetchall()]
        
        # If daily has 0 records today, fallback to latest date so charts don't render completely blank
        if not all_dates and period == "daily":
            cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} ORDER BY att_date DESC LIMIT 1")
            latest_row = cur.fetchone()
            if latest_row:
                all_dates = [latest_row['att_date']]
                where_clause = f"WHERE DATE(Attendance_DateTime) = '{latest_row['att_date']}'"
                
        total_sessions = len(all_dates)
        
        # 2. All registered students
        cur.execute("""
            SELECT s.Enrollment_No, s.Name, COALESCE(s.parent_phone, st.parent_phone, '') AS parent_phone 
            FROM student_detail s 
            LEFT JOIN students st ON s.Enrollment_No = st.enrollment_no 
            ORDER BY s.Enrollment_No ASC
        """)
        students = cur.fetchall()
        total_students = len(students)
        
        # 3. Attendance records for this subject within the period
        cur.execute(f"SELECT Enrollment_No, DATE(Attendance_DateTime) as att_date, Status, Attendance_DateTime FROM {table_name} {where_clause}")
        records = cur.fetchall()
        
        # Aggregate student performance
        student_stats = []
        defaulters_count = 0
        total_pct_sum = 0
        
        for s in students:
            enr = s['Enrollment_No']
            s_records = [r for r in records if r['Enrollment_No'] == enr]
            present_dates = set(r['att_date'] for r in s_records if r['Status'] == 'Present')
            late_dates = set(r['att_date'] for r in s_records if r['Status'] == 'Late')
            attended_count = len(present_dates.union(late_dates))
            
            pct = round((attended_count / total_sessions * 100), 1) if total_sessions > 0 else 0
            status = "Eligible" if pct >= 75 else "Shortage"
            if pct < 75:
                defaulters_count += 1
            total_pct_sum += pct
            
            student_stats.append({
                "enrollment_no": enr,
                "name": s['Name'],
                "parent_phone": s.get('parent_phone') or "",
                "attended": attended_count,
                "total_classes": total_sessions,
                "percentage": pct,
                "status": status,
                "missed": max(0, total_sessions - attended_count)
            })
            
        student_stats.sort(key=lambda x: x["percentage"], reverse=True)
        avg_attendance_pct = round(total_pct_sum / total_students, 1) if total_students > 0 else 0
        
        # 4. Daily attendance trend
        daily_trend = []
        for d in all_dates:
            present_in_day = len(set(r['Enrollment_No'] for r in records if r['att_date'] == d and r['Status'] in ('Present', 'Late')))
            absent_in_day = max(0, total_students - present_in_day)
            daily_trend.append({
                "date": d.strftime("%Y-%m-%d"),
                "display_date": d.strftime("%d %b"),
                "day_name": d.strftime("%A"),
                "present": present_in_day,
                "absent": absent_in_day,
                "attendance_rate": round((present_in_day / total_students * 100), 1) if total_students > 0 else 0
            })
            
        # 5. Day-of-Week Bunk / Absenteeism Analysis
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        day_agg = {d: {"present_sum": 0, "possible_sum": 0, "sessions": 0} for d in day_names}
        
        for d in all_dates:
            d_name = d.strftime("%A")
            if d_name in day_agg:
                day_agg[d_name]["sessions"] += 1
                pres_count = len(set(r['Enrollment_No'] for r in records if r['att_date'] == d and r['Status'] in ('Present', 'Late')))
                day_agg[d_name]["present_sum"] += pres_count
                day_agg[d_name]["possible_sum"] += total_students
                
        day_of_week_analysis = []
        peak_bunk_day = "None"
        lowest_att = 101
        
        for d_name in day_names:
            info = day_agg[d_name]
            att_pct = round((info["present_sum"] / info["possible_sum"] * 100), 1) if info["possible_sum"] > 0 else 0
            bunk_pct = round(100 - att_pct, 1) if info["possible_sum"] > 0 else 0
            
            if info["sessions"] > 0 and att_pct < lowest_att:
                lowest_att = att_pct
                peak_bunk_day = d_name
                
            day_of_week_analysis.append({
                "day": d_name,
                "short_day": d_name[:3],
                "sessions": info["sessions"],
                "attendance_pct": att_pct,
                "bunk_pct": bunk_pct
            })
            
        # 6. Overall status counts
        total_present_instances = sum(1 for r in records if r['Status'] == 'Present')
        total_late_instances = sum(1 for r in records if r['Status'] == 'Late')
        total_possible = total_students * total_sessions
        total_absent_instances = max(0, total_possible - total_present_instances - total_late_instances)
        
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "subject": teacher_subject,
            "period": period,
            "kpi": {
                "total_students": total_students,
                "total_sessions": total_sessions,
                "avg_attendance_pct": avg_attendance_pct,
                "defaulters_count": defaulters_count,
                "peak_bunk_day": peak_bunk_day
            },
            "daily_trend": daily_trend,
            "day_of_week": day_of_week_analysis,
            "status_distribution": {
                "present": total_present_instances,
                "absent": total_absent_instances,
                "late": total_late_instances
            },
            "student_stats": student_stats
        }
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/teacher/export_report")
def api_teacher_export_report(
    period: str = "monthly",
    session: Optional[str] = Cookie(None)
):
    """Generate and download a professional Excel attendance register sheet"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        raise HTTPException(status_code=400, detail="Invalid subject")
        
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        now = datetime.datetime.now()
        where_clause = ""
        period_title = "COMPLETE ACADEMIC REGISTER"
        
        if period == "daily":
            where_clause = "WHERE DATE(Attendance_DateTime) = CURDATE()"
            period_title = f"DAILY REPORT ({now.strftime('%d-%b-%Y')})"
        elif period == "weekly":
            where_clause = "WHERE Attendance_DateTime >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"
            period_title = "WEEKLY REPORT (LAST 7 DAYS)"
        elif period == "monthly":
            where_clause = "WHERE Attendance_DateTime >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
            period_title = "MONTHLY REPORT (LAST 30 DAYS)"
            
        cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} {where_clause} ORDER BY att_date ASC")
        dates_rows = cur.fetchall()
        
        # If daily has no records today, fallback to latest date so user receives a real populated sheet
        if not dates_rows and period == "daily":
            cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} ORDER BY att_date DESC LIMIT 1")
            dates_rows = cur.fetchall()
            if dates_rows:
                fallback_date = dates_rows[0]['att_date']
                where_clause = f"WHERE DATE(Attendance_DateTime) = '{fallback_date}'"
                period_title = f"DAILY REPORT ({fallback_date.strftime('%d-%b-%Y')})"
                
        # If still empty, fetch all dates
        if not dates_rows:
            cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} ORDER BY att_date ASC")
            dates_rows = cur.fetchall()
            period_title = "ALL SESSIONS REGISTER"
            where_clause = ""
            
        distinct_dates = [r['att_date'] for r in dates_rows]
        
        # Fetch students
        cur.execute("SELECT Enrollment_No, Name FROM student_detail ORDER BY Enrollment_No ASC")
        students = cur.fetchall()
        
        # Fetch attendance records
        cur.execute(f"SELECT Enrollment_No, DATE(Attendance_DateTime) as att_date, Status FROM {table_name} {where_clause}")
        att_records = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Build Excel Workbook with openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Attendance Register"
        ws.views.sheetView[0].showGridLines = True
        
        # Styles
        navy_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        soft_blue_fill = PatternFill(start_color="3B6EAE", end_color="3B6EAE", fill_type="solid")
        light_meta_fill = PatternFill(start_color="EEF3F8", end_color="EEF3F8", fill_type="solid")
        header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
        summary_fill = PatternFill(start_color="E9ECEF", end_color="E9ECEF", fill_type="solid")
        
        p_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid") # Present green
        a_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid") # Absent red
        l_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid") # Late yellow
        
        thin_border = Border(
            left=Side(style='thin', color='B0BEC5'),
            right=Side(style='thin', color='B0BEC5'),
            top=Side(style='thin', color='B0BEC5'),
            bottom=Side(style='thin', color='B0BEC5')
        )
        
        total_cols = max(7, 3 + len(distinct_dates) + 4)
        end_col_letter = get_column_letter(total_cols)
        
        # Title Banner (Rows 1-3)
        ws.merge_cells(f"A1:{end_col_letter}1")
        ws["A1"] = "🎓 VISION MARK — SMART ATTENDANCE REGISTER"
        ws["A1"].font = Font(name="Segoe UI", size=15, bold=True, color="FFFFFF")
        ws["A1"].fill = navy_fill
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 36
        
        ws.merge_cells(f"A2:{end_col_letter}2")
        ws["A2"] = f"Subject: {teacher_subject}   |   Faculty: {user.get('name', user.get('username'))}   |   Class: {user.get('class', 'All Batches')}"
        ws["A2"].font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        ws["A2"].fill = soft_blue_fill
        ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 24
        
        ws.merge_cells(f"A3:{end_col_letter}3")
        ws["A3"] = f"Report Range: {period_title}   |   Generated On: {now.strftime('%d-%b-%Y %I:%M %p')}"
        ws["A3"].font = Font(name="Segoe UI", size=10, italic=True, color="333333")
        ws["A3"].fill = light_meta_fill
        ws["A3"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[3].height = 20
        
        # Separator Row 4
        ws.row_dimensions[4].height = 8
        
        # Table Header (Row 5)
        headers = ["S.No", "Enrollment No", "Student Name"]
        for d in distinct_dates:
            headers.append(d.strftime("%d/%m"))
        headers.extend(["Present (P)", "Total Held", "Attendance %", "Status (<75%)"])
        
        for col_idx, h_text in enumerate(headers, 1):
            cell = ws.cell(row=5, column=col_idx, value=h_text)
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
        ws.row_dimensions[5].height = 28
        
        # Data Rows (Row 6 onwards)
        current_row = 6
        date_present_counts = {d: 0 for d in distinct_dates}
        
        for idx, student in enumerate(students, 1):
            enr = student["Enrollment_No"]
            s_name = student["Name"]
            
            c1 = ws.cell(row=current_row, column=1, value=idx)
            c2 = ws.cell(row=current_row, column=2, value=enr)
            c3 = ws.cell(row=current_row, column=3, value=s_name)
            
            for c in [c1, c2]:
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.border = thin_border
                c.font = Font(name="Segoe UI", size=10)
            c3.alignment = Alignment(horizontal="left", vertical="center")
            c3.border = thin_border
            c3.font = Font(name="Segoe UI", size=10, bold=True)
            
            student_present_count = 0
            for d_idx, d in enumerate(distinct_dates, 4):
                records_on_date = [r for r in att_records if r["Enrollment_No"] == enr and r["att_date"] == d]
                cell = ws.cell(row=current_row, column=d_idx)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border
                
                if any(r["Status"] == "Present" for r in records_on_date):
                    cell.value = "P"
                    cell.fill = p_fill
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="155724")
                    student_present_count += 1
                    date_present_counts[d] += 1
                elif any(r["Status"] == "Late" for r in records_on_date):
                    cell.value = "L"
                    cell.fill = l_fill
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="856404")
                    student_present_count += 1
                    date_present_counts[d] += 1
                else:
                    cell.value = "A"
                    cell.fill = a_fill
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="721C24")
                    
            total_sessions = len(distinct_dates)
            pct = round((student_present_count / total_sessions * 100), 1) if total_sessions > 0 else 0
            is_eligible = pct >= 75
            
            col_p = 4 + len(distinct_dates)
            cP = ws.cell(row=current_row, column=col_p, value=student_present_count)
            cT = ws.cell(row=current_row, column=col_p + 1, value=total_sessions)
            cRate = ws.cell(row=current_row, column=col_p + 2, value=f"{pct}%")
            cStat = ws.cell(row=current_row, column=col_p + 3, value="ELIGIBLE" if is_eligible else "SHORTAGE (<75%)")
            
            for c in [cP, cT, cRate]:
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.border = thin_border
                c.font = Font(name="Segoe UI", size=10, bold=True)
                
            cStat.alignment = Alignment(horizontal="center", vertical="center")
            cStat.border = thin_border
            if is_eligible:
                cStat.fill = p_fill
                cStat.font = Font(name="Segoe UI", size=10, bold=True, color="155724")
            else:
                cStat.fill = a_fill
                cStat.font = Font(name="Segoe UI", size=10, bold=True, color="721C24")
                
            ws.row_dimensions[current_row].height = 22
            current_row += 1
            
        # Summary Row at Bottom
        ws.merge_cells(f"A{current_row}:C{current_row}")
        sum_label = ws.cell(row=current_row, column=1, value="TOTAL PRESENT COUNT")
        sum_label.font = Font(name="Segoe UI", size=10, bold=True, color="1B365D")
        sum_label.fill = summary_fill
        sum_label.alignment = Alignment(horizontal="center", vertical="center")
        sum_label.border = thin_border
        
        ws.cell(row=current_row, column=2).border = thin_border
        ws.cell(row=current_row, column=3).border = thin_border
        
        for d_idx, d in enumerate(distinct_dates, 4):
            c = ws.cell(row=current_row, column=d_idx, value=date_present_counts[d])
            c.font = Font(name="Segoe UI", size=10, bold=True, color="1B365D")
            c.fill = summary_fill
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border
            
        col_p = 4 + len(distinct_dates)
        for end_idx in range(col_p, total_cols + 1):
            c = ws.cell(row=current_row, column=end_idx, value="—")
            c.fill = summary_fill
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border
        ws.row_dimensions[current_row].height = 24
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 8
        ws.column_dimensions['B'].width = 18
        ws.column_dimensions['C'].width = 24
        for d_idx in range(4, 4 + len(distinct_dates)):
            ws.column_dimensions[get_column_letter(d_idx)].width = 11
        ws.column_dimensions[get_column_letter(4 + len(distinct_dates))].width = 14
        ws.column_dimensions[get_column_letter(5 + len(distinct_dates))].width = 14
        ws.column_dimensions[get_column_letter(6 + len(distinct_dates))].width = 15
        ws.column_dimensions[get_column_letter(7 + len(distinct_dates))].width = 20
        
        # Stream response
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        safe_sub = "".join(c for c in teacher_subject if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        filename = f"Attendance_{safe_sub}_{period}_{now.strftime('%Y%m%d')}.xlsx"
        
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teacher/export_student_report")
def api_teacher_export_student_report(
    enrollment_no: str,
    session: Optional[str] = Cookie(None)
):
    """Generate and download a comprehensive single-student attendance & entry audit Excel report (.xlsx)"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
    if not table_name:
        raise HTTPException(status_code=400, detail="Invalid subject")
        
    enrollment_no = enrollment_no.strip().upper()
    
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # 1. Fetch student info
        cur.execute("""
            SELECT s.Enrollment_No, s.Name, COALESCE(s.parent_phone, st.parent_phone, '') AS parent_phone, st.class
            FROM student_detail s
            LEFT JOIN students st ON s.Enrollment_No = st.enrollment_no
            WHERE s.Enrollment_No = %s
        """, (enrollment_no,))
        student_info = cur.fetchone()
        
        if not student_info:
            cur.execute("SELECT enrollment_no AS Enrollment_No, name AS Name, parent_phone, class FROM students WHERE enrollment_no = %s", (enrollment_no,))
            student_info = cur.fetchone()
            
        student_name = student_info['Name'] if student_info else enrollment_no
        parent_phone = student_info.get('parent_phone') if student_info else ''
        student_class = student_info.get('class') or 'B.Tech CS / All Batches'
        
        # 2. Fetch all distinct lecture dates conducted for this subject
        cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {table_name} ORDER BY att_date ASC")
        all_dates = [r['att_date'] for r in cur.fetchall()]
        total_sessions = len(all_dates)
        
        # 3. Fetch all attendance entries for this specific student in this subject
        cur.execute(f"""
            SELECT id, Enrollment_No, Attendance_DateTime, Status, recognized_by, confidence, image_path, remarks 
            FROM {table_name} 
            WHERE Enrollment_No = %s 
            ORDER BY Attendance_DateTime ASC
        """, (enrollment_no,))
        student_records = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Group records by date (first entry on that date)
        date_records_map = {}
        for r in student_records:
            d = r['Attendance_DateTime'].date()
            if d not in date_records_map:
                date_records_map[d] = r
                
        present_count = sum(1 for d in all_dates if d in date_records_map and date_records_map[d]['Status'] == 'Present')
        late_count = sum(1 for d in all_dates if d in date_records_map and date_records_map[d]['Status'] == 'Late')
        attended_count = present_count + late_count
        absent_count = max(0, total_sessions - attended_count)
        pct = round((attended_count / total_sessions * 100), 1) if total_sessions > 0 else 0
        is_eligible = pct >= 75
        eligibility = "ELIGIBLE (Passing Criteria ≥ 75%)" if is_eligible else "SHORTAGE DEFAULTER (Below 75%)"
        
        # Build Excel Workbook using openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Student Attendance Audit"
        ws.views.sheetView[0].showGridLines = True
        
        now = datetime.datetime.now()
        
        # Color palettes
        navy_dark = PatternFill(start_color="0F243E", end_color="0F243E", fill_type="solid")
        navy_blue = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        card_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
        th_fill = PatternFill(start_color="2A4365", end_color="2A4365", fill_type="solid")
        
        green_fill = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
        yellow_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
        red_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
        
        green_font = Font(name="Segoe UI", size=10, bold=True, color="065F46")
        yellow_font = Font(name="Segoe UI", size=10, bold=True, color="92400E")
        red_font = Font(name="Segoe UI", size=10, bold=True, color="991B1B")
        
        thin_border = Border(
            left=Side(style='thin', color='CBD5E1'),
            right=Side(style='thin', color='CBD5E1'),
            top=Side(style='thin', color='CBD5E1'),
            bottom=Side(style='thin', color='CBD5E1')
        )
        
        # Title Banner (Row 1-2)
        ws.merge_cells("A1:H1")
        ws["A1"] = "🎓 VISION MARK — INDIVIDUAL STUDENT ATTENDANCE & IN-TIME AUDIT REPORT"
        ws["A1"].font = Font(name="Segoe UI", size=13, bold=True, color="FFFFFF")
        ws["A1"].fill = navy_dark
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 36
        
        ws.merge_cells("A2:H2")
        ws["A2"] = f"Subject: {teacher_subject}   |   Faculty: {user.get('name', user.get('username'))}   |   Report Date: {now.strftime('%d-%b-%Y %I:%M %p')}"
        ws["A2"].font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        ws["A2"].fill = navy_blue
        ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 24
        
        ws.row_dimensions[3].height = 6
        
        # Row 4: Student Details
        ws["A4"] = "Student Name:"
        ws["B4"] = student_name
        ws["C4"] = "Enrollment No:"
        ws["D4"] = enrollment_no
        ws["E4"] = "Class / Batch:"
        ws["F4"] = student_class
        ws["G4"] = "Parent Mobile:"
        ws["H4"] = parent_phone if parent_phone else "N/A"
        
        # Row 5: Attendance Summary Metrics
        ws["A5"] = "Total Lectures:"
        ws["B5"] = total_sessions
        ws["C5"] = "Classes Attended:"
        ws["D5"] = attended_count
        ws["E5"] = "Classes Missed:"
        ws["F5"] = absent_count
        ws["G5"] = "Attendance Rate:"
        ws["H5"] = f"{pct}%"
        
        for r_idx in range(4, 6):
            ws.row_dimensions[r_idx].height = 22
            for col_idx in range(1, 9):
                cell = ws.cell(row=r_idx, column=col_idx)
                cell.border = thin_border
                if col_idx % 2 == 1:
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="475569")
                    cell.fill = card_fill
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="0F172A")
                    cell.alignment = Alignment(horizontal="left", vertical="center")
        
        # Row 6: University Criteria Status Banner
        ws.merge_cells("A6:H6")
        status_banner_text = f"ACADEMIC ELIGIBILITY STATUS:  {eligibility}  (Attended {attended_count} of {total_sessions} classes — {pct}%)"
        ws["A6"] = status_banner_text
        ws["A6"].alignment = Alignment(horizontal="center", vertical="center")
        ws["A6"].font = Font(name="Segoe UI", size=11, bold=True, color="065F46" if is_eligible else "991B1B")
        ws["A6"].fill = green_fill if is_eligible else red_fill
        ws.row_dimensions[6].height = 26
        for col_idx in range(1, 9):
            ws.cell(row=6, column=col_idx).border = thin_border
            
        ws.row_dimensions[7].height = 10
        
        # Table Headers (Row 8)
        table_headers = [
            "S.No",
            "Lecture Date",
            "Day of Week",
            "Exact Entry Time (In-Time)",
            "Attendance Status",
            "Biometric Verification Method",
            "AI Confidence %",
            "Official Remarks / Audit Note"
        ]
        
        ws.row_dimensions[8].height = 28
        for col_idx, h_text in enumerate(table_headers, 1):
            cell = ws.cell(row=8, column=col_idx, value=h_text)
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
            cell.fill = th_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border
            
        # Table Rows (Row 9 onwards)
        row_idx = 9
        for s_idx, d in enumerate(all_dates, 1):
            ws.row_dimensions[row_idx].height = 22
            rec = date_records_map.get(d)
            
            day_name = d.strftime("%A")
            date_str = d.strftime("%d-%b-%Y")
            
            if rec:
                status_str = rec['Status']
                exact_time = rec['Attendance_DateTime'].strftime("%I:%M:%S %p")
                method_raw = str(rec.get('recognized_by') or 'AI Face')
                method = "AI Facial Biometric (ArcFace 512-d)" if 'ai' in method_raw.lower() or 'auto' in method_raw.lower() else rec.get('recognized_by', "Faculty Override")
                conf_val = rec.get('confidence')
                conf_str = f"{(float(conf_val) * 100):.1f}%" if conf_val is not None else "100.0%"
                remarks = rec.get('remarks') or ("Verified Present via Live Classroom Camera" if status_str == 'Present' else "Marked Late Entry by Faculty")
            else:
                status_str = "Absent"
                exact_time = "—"
                method = "Unrecognized / Not in Class"
                conf_str = "—"
                remarks = "Absent — Student was not detected in lecture"
                
            ws.cell(row=row_idx, column=1, value=s_idx).alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(row=row_idx, column=2, value=date_str).alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(row=row_idx, column=3, value=day_name).alignment = Alignment(horizontal="center", vertical="center")
            
            time_cell = ws.cell(row=row_idx, column=4, value=exact_time)
            time_cell.alignment = Alignment(horizontal="center", vertical="center")
            if exact_time != "—":
                time_cell.font = Font(name="Segoe UI", size=10, bold=True, color="1B365D")
                
            status_cell = ws.cell(row=row_idx, column=5, value=status_str)
            status_cell.alignment = Alignment(horizontal="center", vertical="center")
            if status_str == 'Present':
                status_cell.fill = green_fill
                status_cell.font = green_font
            elif status_str == 'Late':
                status_cell.fill = yellow_fill
                status_cell.font = yellow_font
            else:
                status_cell.fill = red_fill
                status_cell.font = red_font
                
            ws.cell(row=row_idx, column=6, value=method).alignment = Alignment(horizontal="left", vertical="center")
            ws.cell(row=row_idx, column=7, value=conf_str).alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(row=row_idx, column=8, value=remarks).alignment = Alignment(horizontal="left", vertical="center")
            
            for c in range(1, 9):
                ws.cell(row=row_idx, column=c).border = thin_border
                
            row_idx += 1
            
        # Summary Note
        ws.row_dimensions[row_idx].height = 10
        row_idx += 1
        ws.merge_cells(f"A{row_idx}:H{row_idx}")
        ws[f"A{row_idx}"] = "🔒 This is a system-generated official attendance audit sheet produced by Vision Mark AI Facial Biometrics. Certified University ERP Record."
        ws[f"A{row_idx}"].font = Font(name="Segoe UI", size=9, italic=True, color="64748B")
        ws[f"A{row_idx}"].alignment = Alignment(horizontal="center", vertical="center")
        
        # Column width adjustments
        col_widths = {
            'A': 8,   # S.No
            'B': 16,  # Date
            'C': 15,  # Day
            'D': 25,  # In-Time
            'E': 16,  # Status
            'F': 35,  # Method
            'G': 18,  # Confidence
            'H': 45   # Remarks
        }
        for col_l, width in col_widths.items():
            ws.column_dimensions[col_l].width = width
            
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        safe_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        filename = f"Attendance_Log_{enrollment_no}_{safe_name}.xlsx"
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STUDENT CREDENTIAL MANAGEMENT ====================

@app.post("/api/teacher/add_student")
async def api_teacher_add_student(
    enrollment_no: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    parent_phone: Optional[str] = Form(None),
    update_mode: Optional[str] = Form("new"),
    camera_snapshots: Optional[str] = Form(None),
    photos: List[UploadFile] = File(default=[]),
    session: Optional[str] = Cookie(None)
):
    """
    Teacher registers a new student OR updates an existing student's face dataset.
    For existing students:
      1. Verifies student username and password. If invalid, displays a security alert.
      2. If valid and update_mode == 'new', prompts teacher to choose between:
         - 'overwrite': Deletes old photos & embeddings, extracts fresh embeddings.
         - 'merge': Keeps old photos, adds new photos, re-averages across all embeddings.
      3. Live reloads multi-distance ArcFace model cache.
    """
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    enrollment_no = enrollment_no.strip().upper()
    username = username.strip()
    password = password.strip()
    name = name.strip()
    class_name = class_name.strip()
    parent_phone = parent_phone.strip() if parent_phone else None
    update_mode = (update_mode or "new").strip().lower()
    
    if not enrollment_no or not username or not password or not name:
        return {"success": False, "msg": "All fields (Enrollment No, Name, Username, Password) are required!"}
    
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Check if student exists by enrollment_no
        cur.execute("SELECT * FROM students WHERE enrollment_no = %s", (enrollment_no,))
        existing_student = cur.fetchone()
        
        if existing_student:
            # ==========================================================
            # EXISTING STUDENT WORKFLOW: VERIFY CREDENTIALS FIRST
            # ==========================================================
            if existing_student['username'] != username or existing_student['password'] != password:
                cur.close()
                conn.close()
                return {
                    "success": False,
                    "is_auth_error": True,
                    "student_name": existing_student['name'],
                    "enrollment_no": enrollment_no,
                    "msg": (
                        "❌ Authentication Failed: Username and password do not match this student's account records!\n\n"
                        "🔒 SECURITY NOTICE:\n"
                        "Student credentials protect sensitive biometric face datasets and academic records. "
                        "Never share these credentials with unauthorized persons as it exposes confidential user data. "
                        "Please verify the student's registered credentials from the Enrolled Students list below. "
                        "If credentials have been compromised or observed by someone else, generate/reset a new password immediately."
                    )
                }
            
            # Credentials are valid. Did teacher specify update_mode ('overwrite' or 'merge')?
            if update_mode not in ('overwrite', 'merge'):
                cur.close()
                conn.close()
                return {
                    "success": False,
                    "requires_choice": True,
                    "student_name": existing_student['name'],
                    "enrollment_no": enrollment_no,
                    "current_username": existing_student['username'],
                    "msg": f"Existing student profile verified for {existing_student['name']} ({enrollment_no}). Please choose whether to overwrite old embeddings or merge new photos."
                }
            
            # Update student details
            cur.execute("""
                UPDATE students 
                SET name = %s, class = %s, parent_phone = %s 
                WHERE enrollment_no = %s
            """, (name, class_name, parent_phone, enrollment_no))
            
            cur.execute("""
                UPDATE student_detail 
                SET Name = %s, parent_phone = %s 
                WHERE Enrollment_No = %s
            """, (name, parent_phone, enrollment_no))
            
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')
            folder_name = f"{enrollment_no}_{clean_name}"
            student_folder = os.path.join(DATASET_DIR, folder_name)
            
            if update_mode == 'overwrite':
                # Delete existing embeddings from database
                cur.execute("DELETE FROM student_embeddings WHERE Enrollment_No = %s", (enrollment_no,))
                conn.commit()
                
                # Clear existing photos in folder
                if os.path.exists(student_folder):
                    for fname in os.listdir(student_folder):
                        fpath = os.path.join(student_folder, fname)
                        if os.path.isfile(fpath):
                            try:
                                os.remove(fpath)
                            except Exception:
                                pass
                os.makedirs(student_folder, exist_ok=True)
                
                # Save new photos
                saved_image_paths = []
                img_counter = 1
                for photo in photos:
                    if photo and photo.filename:
                        content = await photo.read()
                        if len(content) > 0:
                            ext = os.path.splitext(photo.filename)[1].lower() or ".jpg"
                            if ext not in ('.jpg', '.jpeg', '.png'):
                                ext = '.jpg'
                            dest_file = os.path.join(student_folder, f"new_{img_counter}{ext}")
                            with open(dest_file, "wb") as f:
                                f.write(content)
                            saved_image_paths.append(dest_file)
                            img_counter += 1
                
                if camera_snapshots:
                    try:
                        snapshots_list = json.loads(camera_snapshots)
                        for snap_str in snapshots_list:
                            if ',' in snap_str:
                                snap_str = snap_str.split(',', 1)[1]
                            img_bytes = base64.b64decode(snap_str)
                            dest_file = os.path.join(student_folder, f"new_snap_{img_counter}.jpg")
                            with open(dest_file, "wb") as f:
                                f.write(img_bytes)
                            saved_image_paths.append(dest_file)
                            img_counter += 1
                    except Exception as snap_err:
                        print(f"[WARN] Failed to process camera snapshots: {snap_err}")
                
                # Generate fresh ArcFace embeddings
                embeddings_generated = 0
                if saved_image_paths:
                    res = register_student_face_dataset(enrollment_no, name, saved_image_paths)
                    embeddings_generated = res.get("total_faces", 0)
                
                cur.close()
                conn.close()
                
                return {
                    "success": True,
                    "mode": "overwrite",
                    "msg": (
                        f"🔄 Face Dataset Successfully Replaced & Updated!\n\n"
                        f"Student: {name} ({enrollment_no})\n"
                        f"Status: Purani embeddings hata kar nayi photos lagayi gayi hain.\n"
                        f"📸 Nayi Photos Stored: {len(saved_image_paths)}\n"
                        f"⚡ Fresh Face Embeddings: {embeddings_generated}\n"
                        f"Live recognition model refreshed instantly!"
                    ),
                    "photos_count": len(saved_image_paths),
                    "embeddings_count": embeddings_generated
                }
            
            else: # update_mode == 'merge'
                os.makedirs(student_folder, exist_ok=True)
                ts = int(time.time())
                saved_image_paths = []
                img_counter = 1
                
                for photo in photos:
                    if photo and photo.filename:
                        content = await photo.read()
                        if len(content) > 0:
                            ext = os.path.splitext(photo.filename)[1].lower() or ".jpg"
                            if ext not in ('.jpg', '.jpeg', '.png'):
                                ext = '.jpg'
                            dest_file = os.path.join(student_folder, f"merge_{ts}_{img_counter}{ext}")
                            with open(dest_file, "wb") as f:
                                f.write(content)
                            saved_image_paths.append(dest_file)
                            img_counter += 1
                
                if camera_snapshots:
                    try:
                        snapshots_list = json.loads(camera_snapshots)
                        for snap_str in snapshots_list:
                            if ',' in snap_str:
                                snap_str = snap_str.split(',', 1)[1]
                            img_bytes = base64.b64decode(snap_str)
                            dest_file = os.path.join(student_folder, f"merge_snap_{ts}_{img_counter}.jpg")
                            with open(dest_file, "wb") as f:
                                f.write(img_bytes)
                            saved_image_paths.append(dest_file)
                            img_counter += 1
                    except Exception as snap_err:
                        print(f"[WARN] Failed to process camera snapshots: {snap_err}")
                
                embeddings_generated = 0
                if saved_image_paths:
                    res = register_student_face_dataset(enrollment_no, name, saved_image_paths)
                    embeddings_generated = res.get("total_faces", 0)
                
                cur.execute("SELECT COUNT(*) as total_embs FROM student_embeddings WHERE Enrollment_No = %s", (enrollment_no,))
                tot_row = cur.fetchone()
                total_embs = tot_row['total_embs'] if tot_row else embeddings_generated
                
                cur.close()
                conn.close()
                
                return {
                    "success": True,
                    "mode": "merge",
                    "msg": (
                        f"✨ Face Dataset Successfully Merged & Accuracy Enhanced!\n\n"
                        f"Student: {name} ({enrollment_no})\n"
                        f"Status: Purani photos ke saath nayi photos jodi gayi hain.\n"
                        f"📸 Nayi Photos Added: {len(saved_image_paths)}\n"
                        f"⚡ Naye Extracted Embeddings: {embeddings_generated}\n"
                        f"🎯 Total Multi-Distance Embeddings: {total_embs}\n"
                        f"Global and multi-distance vector averages recalculated for maximum recognition accuracy!"
                    ),
                    "photos_count": len(saved_image_paths),
                    "embeddings_count": embeddings_generated,
                    "total_embeddings": total_embs
                }

        # ==========================================================
        # NEW STUDENT REGISTRATION WORKFLOW
        # ==========================================================
        # Check if username already taken by another student
        cur.execute("SELECT id FROM students WHERE username = %s", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return {"success": False, "msg": f"❌ Username '{username}' already taken! Please choose another."}
        
        # Ensure student exists in student_detail
        cur.execute("SELECT Enrollment_No FROM student_detail WHERE Enrollment_No = %s", (enrollment_no,))
        if not cur.fetchone():
            cur.execute("INSERT INTO student_detail (Enrollment_No, Name, parent_phone) VALUES (%s, %s, %s)", (enrollment_no, name, parent_phone))
        else:
            cur.execute("UPDATE student_detail SET parent_phone = %s WHERE Enrollment_No = %s", (parent_phone, enrollment_no))
            
        # Insert into students table
        cur.execute("""
            INSERT INTO students (username, password, name, enrollment_no, class, parent_phone)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (username, password, name, enrollment_no, class_name, parent_phone))
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Create Dataset Folder & Save Student Photos
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')
        folder_name = f"{enrollment_no}_{clean_name}"
        student_folder = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(student_folder, exist_ok=True)
        
        saved_image_paths = []
        img_counter = 1
        
        # Process uploaded files
        for photo in photos:
            if photo and photo.filename:
                content = await photo.read()
                if len(content) > 0:
                    ext = os.path.splitext(photo.filename)[1].lower() or ".jpg"
                    if ext not in ('.jpg', '.jpeg', '.png'):
                        ext = '.jpg'
                    dest_file = os.path.join(student_folder, f"upload_{img_counter}{ext}")
                    with open(dest_file, "wb") as f:
                        f.write(content)
                    saved_image_paths.append(dest_file)
                    img_counter += 1
                    
        # Process webcam snapshots (base64)
        if camera_snapshots:
            try:
                snapshots_list = json.loads(camera_snapshots)
                for snap_str in snapshots_list:
                    if ',' in snap_str:
                        snap_str = snap_str.split(',', 1)[1]
                    img_bytes = base64.b64decode(snap_str)
                    dest_file = os.path.join(student_folder, f"snap_{img_counter}.jpg")
                    with open(dest_file, "wb") as f:
                        f.write(img_bytes)
                    saved_image_paths.append(dest_file)
                    img_counter += 1
            except Exception as snap_err:
                print(f"[WARN] Failed to process camera snapshots: {snap_err}")
                
        # Auto-generate face embeddings using ArcFace
        embeddings_generated = 0
        if saved_image_paths:
            try:
                res = register_student_face_dataset(enrollment_no, name, saved_image_paths)
                embeddings_generated = res.get("total_faces", 0)
            except Exception as emb_err:
                print(f"[ERROR] Face embedding extraction failed: {emb_err}")
                
        folder_msg = f"\n📁 Dataset Folder: Dataset/{folder_name}\n📸 Saved Photos: {len(saved_image_paths)}\n⚡ Face Embeddings: {embeddings_generated} (Live synced with recognition engine!)" if saved_image_paths else ""
        
        return {
            "success": True, 
            "msg": f"✅ Student registered successfully!\n\nName: {name}\nEnrollment: {enrollment_no}\nUsername: {username}\nClass: {class_name}{folder_msg}",
            "photos_count": len(saved_image_paths),
            "embeddings_count": embeddings_generated
        }
    except Exception as e:
        return {"success": False, "msg": f"Database error: {str(e)}"}
    except Exception as e:
        return {"success": False, "msg": f"Database error: {str(e)}"}

@app.get("/api/video_feed")
def api_video_feed():
    """Live MJPEG video stream from server webcam to browser"""
    def frame_generator():
        global recog
        while recog and getattr(recog, 'running', False):
            frame = recog.get_current_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/webcam_status")
def api_webcam_status():
    global recog
    is_running = bool(recog and getattr(recog, 'running', False))
    return {"is_running": is_running}

@app.get("/api/teacher/students_list")
def api_teacher_students_list(session: Optional[str] = Cookie(None)):
    """Teacher views all students with credentials"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, username, password, name, enrollment_no, class, parent_phone 
            FROM students 
            ORDER BY enrollment_no ASC
        """)
        students = cur.fetchall()
        cur.close()
        conn.close()
        return {"success": True, "students": students}
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.post("/api/teacher/reset_student_password")
def api_teacher_reset_student_password(
    student_id: int = Form(...),
    new_password: str = Form(...),
    session: Optional[str] = Cookie(None)
):
    """Teacher resets a student's password"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("UPDATE students SET password = %s WHERE id = %s", (new_password, student_id))
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "msg": "✅ Student password updated successfully!"}
    except Exception as e:
        return {"success": False, "msg": str(e)}

# ==================== STUDENT PROFILE & ANALYTICS ====================

@app.get("/api/student/profile")
def api_student_profile(session: Optional[str] = Cookie(None)):
    """Get rich student profile and cross-subject performance analytics"""
    user = get_session(session)
    if not user or user.get("role") != "student":
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    enrollment_no = user.get("enrollment_no")
    if not enrollment_no:
        return {"success": False, "msg": "Student enrollment number not found in session"}
        
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Fetch student login details
        cur.execute("SELECT * FROM students WHERE username = %s", (user.get("username"),))
        student_info = cur.fetchone()
        
        # Calculate subject-wise attendance for canonical subjects only
        subject_breakdown = []
        total_attended_overall = 0
        total_classes_overall = 0
        
        for subject_name, table_name in CANONICAL_SUBJECTS:
            try:
                # Total sessions held
                cur.execute(f"SELECT COUNT(DISTINCT DATE(Attendance_DateTime)) as total_held FROM {table_name}")
                held_res = cur.fetchone()
                total_held = held_res['total_held'] if held_res else 0
                
                # Attended sessions
                cur.execute(f"""
                    SELECT COUNT(DISTINCT DATE(Attendance_DateTime)) as attended 
                    FROM {table_name} 
                    WHERE Enrollment_No = %s AND Status IN ('Present', 'Late')
                """, (enrollment_no,))
                att_res = cur.fetchone()
                attended = att_res['attended'] if att_res else 0
                
                pct = round((attended / total_held * 100), 1) if total_held > 0 else 0
                total_attended_overall += attended
                total_classes_overall += total_held
                subject_breakdown.append({
                    "subject": subject_name,
                    "attended": attended,
                    "total": total_held,
                    "percentage": pct,
                    "status": "Safe" if pct >= 75 else "Shortage"
                })
            except Exception as e:
                print(f"[WARN] Failed fetching profile breakdown for {table_name}: {e}")
                continue
                
        cur.close()
        conn.close()
        
        overall_pct = round((total_attended_overall / total_classes_overall * 100), 1) if total_classes_overall > 0 else 0
        
        # How many classes needed to reach 75%
        # (attended + x) / (total + x) >= 0.75  =>  0.25*x >= 0.75*total - attended
        needed_classes = 0
        safe_margin = 0
        if total_classes_overall > 0:
            if overall_pct < 75:
                needed = int(np.ceil((0.75 * total_classes_overall - total_attended_overall) / 0.25))
                needed_classes = max(0, needed)
            else:
                # safe margin = number of upcoming classes student could miss while staying >= 75%
                margin = int((total_attended_overall - 0.75 * total_classes_overall) / 0.75)
                safe_margin = max(0, margin)
                
        return {
            "success": True,
            "profile": {
                "name": student_info.get("name") if student_info else user.get("name", "Student"),
                "username": student_info.get("username") if student_info else user.get("username"),
                "enrollment_no": enrollment_no,
                "class": student_info.get("class") if student_info else user.get("class", "Computer Science"),
                "parent_phone": student_info.get("parent_phone") or "",
                "status": "Exam Eligible" if overall_pct >= 75 else "Attendance Shortage",
                "overall_percentage": overall_pct,
                "total_attended": total_attended_overall,
                "total_classes": total_classes_overall,
                "needed_classes": needed_classes,
                "safe_margin": safe_margin
            },
            "subject_breakdown": subject_breakdown
        }
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/student/photo/{enrollment_no}")
def api_student_photo(enrollment_no: str):
    """Serve student profile photo from Dataset directory or return clean SVG avatar"""
    enrollment_no = enrollment_no.strip().upper()
    
    # Check Dataset directory for student photos
    if os.path.exists(DATASET_DIR):
        for entry in os.listdir(DATASET_DIR):
            full_path = os.path.join(DATASET_DIR, entry)
            if os.path.isdir(full_path) and (entry.upper().startswith(enrollment_no) or enrollment_no in entry.upper()):
                # Find first image in directory
                for fname in sorted(os.listdir(full_path)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_file = os.path.join(full_path, fname)
                        if os.path.isfile(img_file) and os.path.getsize(img_file) > 0:
                            media = "image/png" if fname.lower().endswith('.png') else "image/jpeg"
                            return FileResponse(img_file, media_type=media)
                            
    # Fallback SVG Avatar
    initial = enrollment_no[-2:] if len(enrollment_no) >= 2 else "ST"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#1B365D;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#3B82F6;stop-opacity:1" />
        </linearGradient>
      </defs>
      <circle cx="60" cy="60" r="58" fill="url(#grad)" stroke="#FFFFFF" stroke-width="4"/>
      <text x="50%" y="54%" font-family="Segoe UI, sans-serif" font-size="36" font-weight="bold" fill="#FFFFFF" text-anchor="middle" dominant-baseline="middle">{initial}</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/api/student/export_my_report")
def api_student_export_my_report(session: Optional[str] = Cookie(None)):
    """Generate and download student's comprehensive attendance report (.xlsx)"""
    user = get_session(session)
    if not user or user.get("role") != "student":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    enrollment_no = user.get("enrollment_no")
    if not enrollment_no:
        raise HTTPException(status_code=400, detail="Student enrollment number not found")
        
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Student info
        cur.execute("""
            SELECT s.Enrollment_No, s.Name, COALESCE(s.parent_phone, st.parent_phone, '') AS parent_phone, st.class
            FROM student_detail s
            LEFT JOIN students st ON s.Enrollment_No = st.enrollment_no
            WHERE s.Enrollment_No = %s
        """, (enrollment_no,))
        student_info = cur.fetchone()
        
        if not student_info:
            cur.execute("SELECT enrollment_no AS Enrollment_No, name AS Name, parent_phone, class FROM students WHERE enrollment_no = %s", (enrollment_no,))
            student_info = cur.fetchone()
            
        student_name = student_info['Name'] if student_info else user.get('name', enrollment_no)
        parent_phone = student_info.get('parent_phone') if student_info else ''
        student_class = student_info.get('class') or user.get('class', 'B.Tech CS - 7th Sem')
        
        # Collect subject breakdown
        subject_stats = []
        all_detailed_records = []
        total_held_all = 0
        total_attended_all = 0
        
        for sub_name, tbl_name in CANONICAL_SUBJECTS:
            # Total held
            cur.execute(f"SELECT DISTINCT DATE(Attendance_DateTime) as att_date FROM {tbl_name} ORDER BY att_date ASC")
            held_dates = [r['att_date'] for r in cur.fetchall()]
            held_count = len(held_dates)
            
            # Student records
            cur.execute(f"""
                SELECT id, Attendance_DateTime, Status, recognized_by, confidence, remarks 
                FROM {tbl_name}
                WHERE Enrollment_No = %s
                ORDER BY Attendance_DateTime ASC
            """, (enrollment_no,))
            records = cur.fetchall()
            
            att_dates = set(r['Attendance_DateTime'].date() for r in records if r['Status'] in ('Present', 'Late'))
            attended_count = len(att_dates)
            missed_count = max(0, held_count - attended_count)
            pct = round((attended_count / held_count * 100), 1) if held_count > 0 else 0
            
            total_held_all += held_count
            total_attended_all += attended_count
            
            subject_stats.append({
                "subject": sub_name,
                "held": held_count,
                "attended": attended_count,
                "missed": missed_count,
                "percentage": pct,
                "status": "ELIGIBLE" if pct >= 75 else "SHORTAGE (<75%)"
            })
            
            # Map detailed records
            for r in records:
                method_raw = str(r.get('recognized_by') or 'AI Face')
                method = "AI Biometric (ArcFace)" if ('ai' in method_raw.lower() or 'auto' in method_raw.lower()) else method_raw
                all_detailed_records.append({
                    "date": r['Attendance_DateTime'].date(),
                    "datetime": r['Attendance_DateTime'],
                    "subject": sub_name,
                    "status": r['Status'],
                    "method": method,
                    "confidence": r.get('confidence'),
                    "remarks": r.get('remarks') or ("Verified via Biometric Camera" if r['Status'] == 'Present' else "Marked Late Entry")
                })
                
        cur.close()
        conn.close()
        
        overall_pct = round((total_attended_all / total_held_all * 100), 1) if total_held_all > 0 else 0
        is_overall_eligible = overall_pct >= 75
        now = datetime.datetime.now()
        
        # Build Excel Workbook
        wb = openpyxl.Workbook()
        
        # Sheet 1: Academic Summary
        ws1 = wb.active
        ws1.title = "Academic Attendance Summary"
        ws1.views.sheetView[0].showGridLines = True
        
        # Styles
        navy_dark = PatternFill(start_color="0F243E", end_color="0F243E", fill_type="solid")
        navy_blue = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
        soft_card = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        th_fill = PatternFill(start_color="243B53", end_color="243B53", fill_type="solid")
        p_fill = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
        a_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
        summary_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
        
        p_font = Font(name="Segoe UI", size=10, bold=True, color="065F46")
        a_font = Font(name="Segoe UI", size=10, bold=True, color="991B1B")
        
        thin_border = Border(
            left=Side(style='thin', color='CBD5E1'),
            right=Side(style='thin', color='CBD5E1'),
            top=Side(style='thin', color='CBD5E1'),
            bottom=Side(style='thin', color='CBD5E1')
        )
        
        # Title
        ws1.merge_cells("A1:G1")
        ws1["A1"] = "🎓 MEWAR UNIVERSITY — STUDENT ATTENDANCE AUDIT REPORT"
        ws1["A1"].font = Font(name="Segoe UI", size=14, bold=True, color="FFFFFF")
        ws1["A1"].fill = navy_dark
        ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws1.row_dimensions[1].height = 36
        
        ws1.merge_cells("A2:G2")
        ws1["A2"] = f"Official Academic Transcript | Generated On: {now.strftime('%d-%b-%Y %I:%M %p')}"
        ws1["A2"].font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        ws1["A2"].fill = navy_blue
        ws1["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws1.row_dimensions[2].height = 24
        
        ws1.row_dimensions[3].height = 8
        
        # Student Info Grid (Rows 4-5)
        ws1["A4"] = "Student Name:"
        ws1["B4"] = student_name
        ws1["C4"] = "Enrollment No:"
        ws1["D4"] = enrollment_no
        ws1["E4"] = "Class / Semester:"
        ws1["F4"] = student_class
        ws1["G4"] = parent_phone if parent_phone else "N/A"
        
        ws1["A5"] = "Total Sessions:"
        ws1["B5"] = total_held_all
        ws1["C5"] = "Attended Sessions:"
        ws1["D5"] = total_attended_all
        ws1["E5"] = "Missed Sessions:"
        ws1["F5"] = max(0, total_held_all - total_attended_all)
        ws1["G5"] = f"{overall_pct}%"
        
        for r_idx in range(4, 6):
            ws1.row_dimensions[r_idx].height = 22
            for col_idx in range(1, 8):
                cell = ws1.cell(row=r_idx, column=col_idx)
                cell.border = thin_border
                if col_idx in (1, 3, 5):
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="475569")
                    cell.fill = soft_card
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.font = Font(name="Segoe UI", size=9, bold=True, color="0F172A")
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                    
        # Overall Eligibility Banner (Row 6)
        ws1.merge_cells("A6:G6")
        eligibility_label = "EXAM ELIGIBLE (Overall Attendance ≥ 75%)" if is_overall_eligible else "SHORTAGE WARNING (Overall Attendance < 75%)"
        ws1["A6"] = f"OVERALL STATUS: {eligibility_label} — {total_attended_all}/{total_held_all} Sessions ({overall_pct}%)"
        ws1["A6"].font = Font(name="Segoe UI", size=11, bold=True, color="065F46" if is_overall_eligible else "991B1B")
        ws1["A6"].fill = p_fill if is_overall_eligible else a_fill
        ws1["A6"].alignment = Alignment(horizontal="center", vertical="center")
        ws1.row_dimensions[6].height = 26
        for col_idx in range(1, 8):
            ws1.cell(row=6, column=col_idx).border = thin_border
            
        ws1.row_dimensions[7].height = 10
        
        # Subject Breakdown Table Header (Row 8)
        sub_headers = ["S.No", "Subject Name", "Conducted Lectures", "Attended", "Missed", "Attendance %", "Eligibility (<75%)"]
        ws1.row_dimensions[8].height = 26
        for col_idx, h_text in enumerate(sub_headers, 1):
            cell = ws1.cell(row=8, column=col_idx, value=h_text)
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
            cell.fill = th_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
            
        # Data Rows (Row 9 onwards)
        row_cur = 9
        for idx, sub in enumerate(subject_stats, 1):
            ws1.row_dimensions[row_cur].height = 22
            c1 = ws1.cell(row=row_cur, column=1, value=idx)
            c2 = ws1.cell(row=row_cur, column=2, value=sub['subject'])
            c3 = ws1.cell(row=row_cur, column=3, value=sub['held'])
            c4 = ws1.cell(row=row_cur, column=4, value=sub['attended'])
            c5 = ws1.cell(row=row_cur, column=5, value=sub['missed'])
            c6 = ws1.cell(row=row_cur, column=6, value=f"{sub['percentage']}%")
            c7 = ws1.cell(row=row_cur, column=7, value=sub['status'])
            
            for c in [c1, c3, c4, c5, c6]:
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.font = Font(name="Segoe UI", size=10)
                c.border = thin_border
            c2.alignment = Alignment(horizontal="left", vertical="center")
            c2.font = Font(name="Segoe UI", size=10, bold=True)
            c2.border = thin_border
            
            c7.alignment = Alignment(horizontal="center", vertical="center")
            c7.border = thin_border
            if sub['percentage'] >= 75:
                c7.fill = p_fill
                c7.font = p_font
            else:
                c7.fill = a_fill
                c7.font = a_font
            row_cur += 1
            
        # Total Summary Row
        ws1.row_dimensions[row_cur].height = 24
        ws1.merge_cells(f"A{row_cur}:B{row_cur}")
        c_tot = ws1.cell(row=row_cur, column=1, value="OVERALL TOTAL")
        c_tot.fill = summary_fill
        c_tot.font = Font(name="Segoe UI", size=10, bold=True, color="0F172A")
        c_tot.alignment = Alignment(horizontal="center", vertical="center")
        ws1.cell(row=row_cur, column=2).border = thin_border
        
        cH = ws1.cell(row=row_cur, column=3, value=total_held_all)
        cA = ws1.cell(row=row_cur, column=4, value=total_attended_all)
        cM = ws1.cell(row=row_cur, column=5, value=max(0, total_held_all - total_attended_all))
        cPct = ws1.cell(row=row_cur, column=6, value=f"{overall_pct}%")
        cStat = ws1.cell(row=row_cur, column=7, value="ELIGIBLE" if is_overall_eligible else "SHORTAGE")
        
        for c in [c_tot, cH, cA, cM, cPct, cStat]:
            c.border = thin_border
            c.fill = summary_fill
            c.font = Font(name="Segoe UI", size=10, bold=True, color="0F172A")
            c.alignment = Alignment(horizontal="center", vertical="center")
            
        ws1.column_dimensions['A'].width = 8
        ws1.column_dimensions['B'].width = 45
        ws1.column_dimensions['C'].width = 18
        ws1.column_dimensions['D'].width = 14
        ws1.column_dimensions['E'].width = 14
        ws1.column_dimensions['F'].width = 16
        ws1.column_dimensions['G'].width = 22
        
        # Sheet 2: Detailed Session Entries
        ws2 = wb.create_sheet(title="Detailed Session Entries")
        ws2.views.sheetView[0].showGridLines = True
        
        ws2.merge_cells("A1:H1")
        ws2["A1"] = f"SESSION-BY-SESSION ATTENDANCE AUDIT LOG — {student_name} ({enrollment_no})"
        ws2["A1"].font = Font(name="Segoe UI", size=12, bold=True, color="FFFFFF")
        ws2["A1"].fill = navy_dark
        ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws2.row_dimensions[1].height = 32
        
        headers2 = ["S.No", "Date", "Day", "Subject", "Class In-Time", "Status", "Biometric Method / Marked By", "Official Remarks / Reason"]
        ws2.row_dimensions[2].height = 24
        for col_idx, h_text in enumerate(headers2, 1):
            cell = ws2.cell(row=2, column=col_idx, value=h_text)
            cell.font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
            cell.fill = th_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
            
        all_detailed_records.sort(key=lambda x: x['datetime'], reverse=True)
        
        r2_cur = 3
        for idx, rec in enumerate(all_detailed_records, 1):
            ws2.row_dimensions[r2_cur].height = 20
            d = rec['datetime']
            
            c1 = ws2.cell(row=r2_cur, column=1, value=idx)
            c2 = ws2.cell(row=r2_cur, column=2, value=d.strftime("%d-%b-%Y"))
            c3 = ws2.cell(row=r2_cur, column=3, value=d.strftime("%A"))
            c4 = ws2.cell(row=r2_cur, column=4, value=rec['subject'])
            c5 = ws2.cell(row=r2_cur, column=5, value=d.strftime("%I:%M:%S %p"))
            c6 = ws2.cell(row=r2_cur, column=6, value=rec['status'])
            
            method_str = rec.get('method') or "AI Biometric (ArcFace)"
            c7 = ws2.cell(row=r2_cur, column=7, value=method_str)
            c8 = ws2.cell(row=r2_cur, column=8, value=rec.get('remarks') or "Verified Present")
            
            for c in [c1, c2, c3, c5]:
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.border = thin_border
                c.font = Font(name="Segoe UI", size=9)
                
            c4.alignment = Alignment(horizontal="left", vertical="center")
            c4.border = thin_border
            c4.font = Font(name="Segoe UI", size=9)
            
            c6.alignment = Alignment(horizontal="center", vertical="center")
            c6.border = thin_border
            if rec['status'] == 'Present':
                c6.fill = p_fill
                c6.font = p_font
            else:
                c6.fill = a_fill
                c6.font = a_font
                
            c7.alignment = Alignment(horizontal="left", vertical="center")
            c7.border = thin_border
            c7.font = Font(name="Segoe UI", size=9)
            
            c8.alignment = Alignment(horizontal="left", vertical="center")
            c8.border = thin_border
            c8.font = Font(name="Segoe UI", size=9)
            
            r2_cur += 1
            
        ws2.column_dimensions['A'].width = 8
        ws2.column_dimensions['B'].width = 16
        ws2.column_dimensions['C'].width = 15
        ws2.column_dimensions['D'].width = 38
        ws2.column_dimensions['E'].width = 18
        ws2.column_dimensions['F'].width = 14
        ws2.column_dimensions['G'].width = 30
        ws2.column_dimensions['H'].width = 45
        
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        clean_stud_name = "".join(c for c in student_name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
        filename = f"My_Attendance_Report_{enrollment_no}_{clean_stud_name}.xlsx"
        
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start_webcam")
def api_start_webcam(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    global recog_thread, current_recognitions, active_camera_teacher
    active_camera_teacher = user
    
    if recog_thread and recog_thread.is_alive():
        return {"success": True, "msg": "Camera is already running"}
    
    current_recognitions.clear()
    
    def run():
        recog.run_webcam(show_window=False)
    
    recog_thread = threading.Thread(target=run, daemon=True)
    recog_thread.start()
    
    return {"success": True, "msg": "Webcam started"}

@app.get("/api/stop_webcam")
def api_stop_webcam(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    global active_camera_teacher
    active_camera_teacher = None
    
    try:
        recog.stop_webcam()
        return {"success": True, "msg": "Stopping webcam"}
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/recent_recognitions")
def api_recent_recognitions(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return {"recognitions": current_recognitions[-20:]}  # Last 20

@app.post("/api/test_image")
async def api_test_image(file: UploadFile = File(...), session: Optional[str] = Cookie(None)):
    """Upload image for face recognition testing"""
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Save uploaded file
        contents = await file.read()
        filename = f"test_{int(datetime.datetime.now().timestamp())}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Process image
        img = cv2.imread(filepath)
        if img is None:
            return {"success": False, "msg": "Invalid image"}
        
        # Detect and recognize faces
        result_img, faces = recog.process_frame(img)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOAD_DIR, result_filename)
        cv2.imwrite(result_path, result_img)
        
        # Extract recognition results
        results = []
        for face in faces:
            if 'embedding' in face and face['embedding'] is not None:
                match, similarity = recog.recognize_face(face['embedding'], face['bbox'])
                results.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'recognized': match is not None,
                    'name': match['name'] if match else 'Unknown',
                    'enrollment_no': match['enrollment_no'] if match else None,
                    'similarity': float(similarity)
                })
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "results": results,
            "result_image": f"/static/uploads/{result_filename}"
        }
        
    except Exception as e:
        return {"success": False, "msg": str(e)}

# Run with: uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
# -----------------------
# Run with: uvicorn backend.server:app --reload
# -----------------------
