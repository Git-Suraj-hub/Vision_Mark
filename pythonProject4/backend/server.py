# backend/server.py
import os
import io
import secrets
import datetime
import threading
import base64
from typing import Optional
from fastapi import FastAPI, Request, Form, Cookie, Response, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mysql.connector
from contextlib import asynccontextmanager
import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env file
load_dotenv()
# Import recognition class
from backend.Recognize2 import AdvancedFaceRecognition

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
CAPTURE_DIR = os.path.join(STATIC_DIR, "captures")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

# Create directories
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# Database helper
def get_db_conn():
    return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )

# Subject to table mapping
SUBJECT_TABLE_MAP = {
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

# Global recognition object
recog = None
recog_thread = None
current_recognitions = []  # Store recent recognitions

# Recognition callback
def recognition_callback(match, similarity, bbox, img_bytes):
    """Called when face is recognized during webcam"""
    try:
        enrollment_no = match.get("enrollment_no")
        name = match.get("name")
        
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        # Get current subject from timetable
        now = datetime.datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        cur.execute("""
            SELECT Subject, Start_Time, End_Time FROM DummyTimeTable
            WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
        """, (current_day, current_time, current_time))
        
        result = cur.fetchone()
        
        if not result:
            print("[INFO] No class running right now")
            cur.close()
            conn.close()
            return
        
        subject = result['Subject']
        start_time = result['Start_Time']
        end_time = result['End_Time']
        
        table_name = SUBJECT_TABLE_MAP.get(subject)
        if not table_name:
            print(f"[WARN] No table found for subject '{subject}'")
            cur.close()
            conn.close()
            return
        
        # Check if already marked in this time slot
        cur.execute(f"""
            SELECT * FROM {table_name}
            WHERE Enrollment_No = %s
            AND DATE(Attendance_DateTime) = CURDATE()
            AND TIME(Attendance_DateTime) BETWEEN %s AND %s
        """, (enrollment_no, start_time, end_time))
        
        already_marked = cur.fetchone()
        
        if already_marked:
            print(f"[INFO] {enrollment_no} already marked for {subject}")
            cur.close()
            conn.close()
            return
        
        # Save captured image
        image_path = None
        if img_bytes:
            fname = f"{enrollment_no}_{int(now.timestamp())}.jpg"
            image_path = os.path.join("captures", fname)
            full_path = os.path.join(CAPTURE_DIR, fname)
            with open(full_path, "wb") as f:
                f.write(img_bytes)
        
        # Mark attendance
        cur.execute(f"""
            INSERT INTO {table_name} 
            (Enrollment_No, Attendance_DateTime, Status, recognized_by, confidence, image_path)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (enrollment_no, now, "Present", "auto_face", float(similarity), image_path))
        
        conn.commit()
        
        # Add to recent recognitions
        current_recognitions.append({
            'enrollment_no': enrollment_no,
            'name': name,
            'subject': subject,
            'time': now.strftime("%H:%M:%S"),
            'confidence': float(similarity),
            'image_path': image_path
        })
        
        # Keep only last 50 recognitions
        if len(current_recognitions) > 50:
            current_recognitions.pop(0)
        
        print(f"[SUCCESS] Attendance marked for {name} ({enrollment_no}) in {subject}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"[ERROR] recognition_callback: {e}")
        import traceback
        traceback.print_exc()

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
            FROM DummyTimeTable
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

@app.get("/api/student/attendance/{enrollment_no}")
def api_student_attendance(enrollment_no: str):
    """Get student attendance across all subjects"""
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        attendance_data = {}
        
        for subject, table_name in SUBJECT_TABLE_MAP.items():
            try:
                cur.execute(f"""
                    SELECT Enrollment_No, Attendance_DateTime, Status, confidence, image_path
                    FROM {table_name}
                    WHERE Enrollment_No = %s
                    ORDER BY Attendance_DateTime DESC
                    LIMIT 100
                """, (enrollment_no,))
                
                records = cur.fetchall()
                attendance_data[subject] = [
                    {
                        'date': r['Attendance_DateTime'].strftime("%Y-%m-%d"),
                        'time': r['Attendance_DateTime'].strftime("%H:%M:%S"),
                        'status': r['Status'],
                        'confidence': r.get('confidence', 1.0),
                        'image_path': r.get('image_path')
                    } for r in records
                ]
            except Exception as e:
                print(f"[ERROR] Getting attendance from {table_name}: {e}")
                attendance_data[subject] = []
        
        cur.close()
        conn.close()
        
        return {"success": True, "attendance": attendance_data}
        
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.get("/api/teacher/attendance")
def api_teacher_attendance(session: Optional[str] = Cookie(None)):
    """Get attendance for teacher's subject only"""
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
        
        # Get all attendance for this subject
        cur.execute(f"""
            SELECT a.*, s.Name 
            FROM {table_name} a
            LEFT JOIN student_detail s ON a.Enrollment_No = s.Enrollment_No
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
                'enrollment_no': r['Enrollment_No'],
                'name': r.get('Name', 'Unknown'),
                'date': r['Attendance_DateTime'].strftime("%Y-%m-%d"),
                'time': r['Attendance_DateTime'].strftime("%H:%M:%S"),
                'status': r['Status'],
                'recognized_by': r.get('recognized_by', 'manual'),
                'confidence': r.get('confidence', 1.0),
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

@app.post("/api/mark_attendance")
def api_mark_attendance(
    enrollment_no: str = Form(...),
    status: str = Form("Present"),
    session: Optional[str] = Cookie(None)
):
    """
    ✅ FIXED: Manual attendance marking with proper validation
    - Checks if teacher's subject matches current class
    - Prevents duplicate marking in same time slot
    - Validates current class is running
    """
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    teacher_subject = user.get("subject")
    teacher_name = user.get("name", user.get("username"))
    
    if not teacher_subject:
        return {"success": False, "msg": "No subject assigned to you"}
    
    try:
        conn = get_db_conn()
        cur = conn.cursor(dictionary=True)
        
        now = datetime.datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        # ============================================
        # STEP 1: Check if ANY class is running now
        # ============================================
        cur.execute("""
            SELECT Subject, Start_Time, End_Time, Teacher, Room 
            FROM DummyTimeTable
            WHERE Day = %s AND Start_Time <= %s AND End_Time >= %s
        """, (current_day, current_time, current_time))
        
        current_class = cur.fetchone()
        
        if not current_class:
            cur.close()
            conn.close()
            return {
                "success": False, 
                "msg": "❌ No class is running right now. Cannot mark attendance."
            }
        
        current_subject = current_class['Subject']
        start_time = current_class['Start_Time']
        end_time = current_class['End_Time']
        
        # ============================================
        # STEP 2: Check if teacher's subject matches current class
        # ============================================
        if teacher_subject != current_subject:
            cur.close()
            conn.close()
            return {
                "success": False,
                "msg": f"❌ Cannot mark attendance!\n\n"
                       f"Current running class: {current_subject}\n"
                       f"Your subject: {teacher_subject}\n\n"
                       f"You can only mark attendance for your own subject.",
                "current_subject": current_subject,
                "teacher_subject": teacher_subject
            }
        
        # ============================================
        # STEP 3: Get the table name for this subject
        # ============================================
        table_name = SUBJECT_TABLE_MAP.get(teacher_subject)
        if not table_name:
            cur.close()
            conn.close()
            return {"success": False, "msg": f"Invalid subject: {teacher_subject}"}
        
        # ============================================
        # STEP 4: Check if student exists
        # ============================================
        cur.execute("""
            SELECT Name FROM student_detail WHERE Enrollment_No = %s
        """, (enrollment_no,))
        
        student = cur.fetchone()
        if not student:
            cur.close()
            conn.close()
            return {
                "success": False,
                "msg": f"❌ Student {enrollment_no} not found in database"
            }
        
        student_name = student['Name']
        
        # ============================================
        # STEP 5: Check for duplicate in SAME time slot
        # ============================================
        cur.execute(f"""
            SELECT * FROM {table_name}
            WHERE Enrollment_No = %s
            AND DATE(Attendance_DateTime) = CURDATE()
            AND TIME(Attendance_DateTime) BETWEEN %s AND %s
        """, (enrollment_no, start_time, end_time))
        
        existing = cur.fetchone()
        
        if existing:
            existing_time = existing['Attendance_DateTime'].strftime("%H:%M:%S")
            existing_method = existing.get('recognized_by', 'unknown')
            
            cur.close()
            conn.close()
            
            return {
                "success": False,
                "msg": f"⚠️ Attendance already marked!\n\n"
                       f"Student: {student_name} ({enrollment_no})\n"
                       f"Subject: {teacher_subject}\n"
                       f"Time: {existing_time}\n"
                       f"Method: {existing_method}\n\n"
                       f"Cannot mark again in the same class period ({start_time} - {end_time})",
                "already_marked": True,
                "existing_record": {
                    "time": existing_time,
                    "method": existing_method
                }
            }
        
        # ============================================
        # STEP 6: Mark attendance
        # ============================================
        cur.execute(f"""
            INSERT INTO {table_name}
            (Enrollment_No, Attendance_DateTime, Status, recognized_by, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (enrollment_no, now, status, "manual", 1.0))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "success": True,
            "msg": f"✅ Attendance marked successfully!\n\n"
                   f"Student: {student_name} ({enrollment_no})\n"
                   f"Subject: {teacher_subject}\n"
                   f"Status: {status}\n"
                   f"Time: {now.strftime('%H:%M:%S')}\n"
                   f"Marked by: {teacher_name} (Manual)",
            "record": {
                "student_name": student_name,
                "enrollment_no": enrollment_no,
                "subject": teacher_subject,
                "status": status,
                "time": now.strftime("%H:%M:%S"),
                "method": "manual"
            }
        }
        
    except Exception as e:
        return {
            "success": False, 
            "msg": f"❌ Database error: {str(e)}"
        }

@app.post("/api/start_webcam")
def api_start_webcam(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    global recog_thread, current_recognitions
    
    if recog_thread and recog_thread.is_alive():
        return {"success": False, "msg": "Already running"}
    
    current_recognitions.clear()
    
    def run():
        recog.run_webcam()
    
    recog_thread = threading.Thread(target=run, daemon=True)
    recog_thread.start()
    
    return {"success": True, "msg": "Webcam started"}

@app.get("/api/stop_webcam")
def api_stop_webcam(session: Optional[str] = Cookie(None)):
    user = get_session(session)
    if not user or user.get("role") != "teacher":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
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
