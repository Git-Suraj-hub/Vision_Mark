# 🎓 Smart Attendance System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> AI-Powered Face Recognition Attendance System with Multi-Distance Detection & Automatic Absent Marking

An advanced automated attendance management system using **ArcFace** and **RetinaFace** for face recognition, supporting multiple distance categories and automatic absent marking.

![System Demo](Demo/Vision_Mark_UI.png)

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Database Setup](#-database-setup)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🎯 Core Features
- **Multi-Distance Face Recognition** - Detects faces from very close to far distances (20px-150px+)
- **Dual Model System** - Uses ArcFace for speed and RetinaFace for accuracy
- **4-Level Recognition Strategy** - Individual embeddings → Adjacent categories → Category averages → Global fallback
- **Automatic Absent Marking** - Auto-marks absent students after class ends (5-minute grace period)
- **Subject-Specific Attendance** - Separate tables for each subject with proper tracking
- **Real-Time Face Recognition** - Live webcam feed with instant detection
- **Duplicate Prevention** - Prevents marking same student twice in same time slot

### 👨‍🎓 Student Features
- Personal attendance dashboard
- Subject-wise attendance view
- Overall attendance statistics
- Attendance percentage calculation
- View captured face images
- Download attendance reports (PDF/Excel/Word)

### 👨‍🏫 Teacher Features
- Subject-specific dashboard (teachers only see their assigned subject)
- Start/Stop face recognition
- Manual attendance marking with validation
- Image upload for face testing
- Real-time recognition feed
- View captured face images
- Automatic absent marking control
- Backfill past absences
- Download class reports (PDF/Excel/Word)
- Student-wise detailed reports

### 🔒 Security Features
- Session-based authentication
- Role-based access control (Student/Teacher)
- Subject-level authorization for teachers
- Duplicate attendance prevention
- Time-based validation
- SQL injection protection

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Auth       │  │  Attendance  │  │   Reports    │     │
│  │  Management  │  │   Marking    │  │  Generation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Face Recognition Engine                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐           │
│  │ ArcFace  │  │RetinaFace│  │Multi-Distance  │           │
│  │ Buffalo-L│  │          │  │  Recognition   │           │
│  └──────────┘  └──────────┘  └────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MySQL Database                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │  Students   │  │  Teachers   │  │ Student      │       │
│  │             │  │             │  │ Detail       │       │
│  └─────────────┘  └─────────────┘  └──────────────┘       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │Student      │  │ DummyTime   │  │Subject-wise  │       │
│  │Embeddings   │  │ Table       │  │ Attendance   │       │
│  └─────────────┘  └─────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.8+** - Core programming language
- **MySQL 8.0+** - Relational database

### Face Recognition
- **InsightFace (ArcFace)** - Primary face recognition model
- **RetinaFace** - High-accuracy face detection
- **OpenCV** - Computer vision operations
- **NumPy** - Numerical computations

### Report Generation
- **ReportLab** - PDF generation
- **OpenPyXL** - Excel generation
- **python-docx** - Word document generation

### Scheduling
- **Schedule** - Background task scheduling for absent marking

### Frontend
- **HTML5/CSS3** - Modern responsive UI
- **JavaScript (Vanilla)** - Client-side interactions
- **Jinja2** - Server-side templating

---

## 📦 Prerequisites

Before installation, ensure you have:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **MySQL 8.0 or higher** ([Download](https://dev.mysql.com/downloads/))
- **Webcam** (for face recognition)
- **Git** ([Download](https://git-scm.com/downloads))
- **4GB+ RAM** (recommended for face recognition models)
- **Good lighting** (for better face detection accuracy)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Git-Suraj-hub/Vision_Mark.git
cd pythonProject4
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
mysql-connector-python==8.2.0
insightface==0.7.3
retinaface==0.0.13
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.4
python-multipart==0.0.6
schedule==1.2.0
reportlab==4.0.7
openpyxl==3.1.2
python-docx==1.1.0
torch==2.1.0
onnxruntime==1.16.3
```

### 4. Install Additional System Dependencies (if needed)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install opencv
```

---

## 🗄️ Database Setup

### 1. Create Database

```bash
mysql -u root -p
```

```sql
CREATE DATABASE face_detection;
USE face_detection;
```

### 2. Run Schema Files

Execute the following SQL files in order:

```bash
# 1. Create base tables
mysql -u root -p face_detection < database/01_base_tables.sql

# 2. Create subject attendance tables
mysql -u root -p face_detection < database/02_subject_tables.sql

# 3. Create embeddings table
mysql -u root -p face_detection < database/03_embeddings_table.sql

# 4. Add category-specific columns
mysql -u root -p face_detection < database/04_category_columns.sql

# 5. Create timetable
mysql -u root -p face_detection < database/05_timetable.sql
```

### 3. Create Sample Data (Optional)

```sql
-- Sample student
INSERT INTO students (username, password, name, enrollment_no, class) 
VALUES ('student1', 'pass123', 'John Doe', 'MUR2300001', 'CSE-A');

-- Sample teacher
INSERT INTO teachers (username, password, name, class, subject) 
VALUES ('teacher1', 'pass123', 'Dr. Smith', 'CSE-A', 'Theory of Computation');

-- Sample timetable entry (24/7 for testing)
INSERT INTO DummyTimeTable (Day, Start_Time, End_Time, Subject, Teacher, Room) 
VALUES ('Monday', '00:00:00', '23:59:59', 'Theory of Computation', 'Dr. Smith', 'A101');
```

---

## ⚙️ Configuration

### 1. Update Database Credentials

Edit the following files with your MySQL credentials:

**backend/server.py:**
```python
def get_db_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YOUR_PASSWORD_HERE",  # Change this
        database="face_detection"
    )
```

**backend/Recognize2.py:**
```python
self.conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YOUR_PASSWORD_HERE",  # Change this
    database="face_detection"
)
```

### 2. Add Logo and Favicon

Place your images in:
```
backend/static/images/logo.png
backend/static/images/favicon.png
```

**Image specifications:**
- Logo: 512x512px, transparent PNG
- Favicon: 32x32px or 64x64px PNG

**Temporary solution:**
```bash
# Use placeholder images
cp demo.png backend/static/images/logo.png
cp demo.png backend/static/images/favicon.png
```

---

## 📸 Generate Face Embeddings

Before using the system, you need to generate face embeddings for students.

### Method 1: From Photos (Batch Processing)

```bash
# 1. Organize photos in Dataset folder:
Dataset/
├── MUR2300001_JohnDoe/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── MUR2300002_JaneSmith/
│   ├── 1.jpg
│   └── 2.jpg
└── ...

# 2. Run embedding generator
python student_Embedding.py

# 3. Calculate category averages
python student_detail_embedding.py
```

### Method 2: Using Webcam (Individual)

```bash
python generate_embeddings.py
# Choose option 3 (Webcam capture)
# Enter enrollment number and name
# Press SPACE to capture
```

---

## 🏃 Running the Application

### Start the Server

```bash
# Development mode (with auto-reload)
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access the Application

Open your browser and navigate to:

```
http://localhost:8000
```

### Default Test Credentials

**Student:**
- Username: `student1`
- Password: `pass123`

**Teacher:**
- Username: `teacher1`
- Password: `pass123`

---

## 📖 Usage Guide

### For Students

1. **Login**
   - Go to http://localhost:8000
   - Click "Student Login"
   - Enter credentials

2. **View Attendance**
   - Dashboard shows overall statistics
   - Click subject tabs to view subject-wise attendance
   - See present/absent records with timestamps


### For Teachers

1. **Login**
   - Go to http://localhost:8000
   - Click "Teacher Login"
   - Enter credentials

2. **Start Face Recognition**
   - Click "Start Recognition" button
   - Webcam will open
   - Students' faces will be detected automatically
   - Attendance marked in real-time

3. **Manual Marking**
   - Enter student enrollment number
   - Select status (Present/Absent)
   - Click "Mark Attendance"

4. **Absent Marking**
   - Click "Mark Absences NOW" for immediate marking
   - Click "Backfill Past Absences" to recover past data

---

## 📁 Project Structure

```
smart-attendance-system/
├── backend/
│   ├── server.py                 # FastAPI server
│   ├── Recognize2.py             # Face recognition engine
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Styles
│   │   ├── images/
│   │   │   ├── logo.png
│   │   │   └── favicon.png
│   │   ├── captures/             # Face captures (auto-created)
│   │   └── uploads/              # Test uploads (auto-created)
│   └── templates/
│       ├── index.html            # Landing page
│       ├── login_student.html
│       ├── login_teacher.html
│       ├── student.html          # Student dashboard
│       └── teacher.html          # Teacher dashboard
├── database/
│   ├── 01_base_tables.sql
│   ├── 02_subject_tables.sql
│   ├── 03_embeddings_table.sql
│   ├── 04_category_columns.sql
│   └── 05_timetable.sql
├── Dataset/                      # Student photos
│   └── ENROLLMENT_NAME/
│       └── *.jpg
├── student_Embedding.py          # Generate embeddings
├── student_detail_embedding.py   # Calculate averages
├── generate_embeddings.py        # Interactive embedding tool
├── absent_marker.py              # Standalone absent marker
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📡 API Documentation

### Authentication

**Login**
```http
POST /api/login
Content-Type: multipart/form-data

role=student&username=student1&password=pass123
```

**Logout**
```http
POST /api/logout
```

### Attendance

**Get Student Attendance**
```http
GET /api/student/attendance/{enrollment_no}
```

**Get Teacher's Subject Attendance**
```http
GET /api/teacher/attendance
Cookie: session={token}
```

**Mark Attendance Manually**
```http
POST /api/mark_attendance
Content-Type: multipart/form-data

enrollment_no=MUR001&status=Present
```

### Recognition

**Start Webcam**
```http
POST /api/start_webcam
Cookie: session={token}
```

**Stop Webcam**
```http
GET /api/stop_webcam
Cookie: session={token}
```

**Test Image**
```http
POST /api/test_image
Content-Type: multipart/form-data

file={image_file}
```

### Absent Marking

**Mark Absences Now**
```http
POST /api/mark_absences_now
Cookie: session={token}
```

**Backfill Absences**
```http
POST /api/backfill_absences
Content-Type: multipart/form-data

days_back=7
```

### Utility

**Get Current Subject**
```http
GET /api/current_subject
```

**Get Current User**
```http
GET /api/current_user
Cookie: session={token}
```

---

## 🔧 Troubleshooting

### Models Not Loading

**Problem:** Face recognition models fail to load

**Solution:**
```bash
# Models download automatically on first run
# Ensure good internet connection
# Models are cached in ~/.insightface/

# If issues persist:
rm -rf ~/.insightface/
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
```

### Webcam Not Opening

**Problem:** Webcam fails to start

**Solution:**
```python
# Try different camera index in Recognize2.py
cv2.VideoCapture(0)  # Change 0 to 1, 2, etc.

# Check camera permissions
# Windows: Settings > Privacy > Camera
# Mac: System Preferences > Security > Camera
# Linux: Check /dev/video* permissions
```

### No Face Detected

**Problem:** Faces not being recognized

**Solution:**
- Ensure good lighting
- Face should be frontal and clearly visible
- Minimum face size: 40x40 pixels
- Check if embeddings exist for student
- Verify camera is working: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`

### Database Connection Error

**Problem:** Cannot connect to MySQL

**Solution:**
```bash
# Check MySQL is running
sudo service mysql status

# Test connection
mysql -u root -p

# Verify database exists
mysql -u root -p -e "SHOW DATABASES LIKE 'face_detection';"
```

### Attendance Not Marking

**Problem:** Automatic attendance marking fails

**Solution:**
- Verify DummyTimeTable has entry for current day/time
- Check teacher's subject matches current class
- Ensure face embeddings exist for student
- Check server logs for errors
- Verify absent marking scheduler is running

### High CPU/Memory Usage

**Problem:** System uses too much resources

**Solution:**
```python
# In Recognize2.py, increase frame skip
skip_frames = 5  # Process every 5th frame instead of 2

# Reduce detection size
self.arcface_app.prepare(ctx_id=0, det_size=(320, 320))  # Instead of (640, 640)
```

---

## 🎯 Performance Optimization

### For Better Accuracy

1. **Good Lighting** - Well-lit environment improves detection
2. **Multiple Photos** - 5+ photos per student at different distances
3. **Quality Photos** - Clear, frontal face images
4. **Regular Updates** - Re-generate embeddings periodically

### For Better Speed

1. **Frame Skipping** - Adjust `skip_frames` in Recognize2.py
2. **Detection Size** - Reduce `det_size` for faster processing
3. **Database Indexing** - Add indexes on frequently queried columns
4. **Hardware** - Use GPU if available (change to `CUDAExecutionProvider`)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 for Python code
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **InsightFace** - Face recognition models
- **RetinaFace** - Face detection
- **FastAPI** - Web framework
- **OpenCV** - Computer vision library
- **ReportLab** - PDF generation

---

## 📞 Support

For issues and questions:

1. Check [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/smart-attendance-system/issues)
3. Create a new issue with:
   - Detailed description
   - Steps to reproduce
   - Error messages/logs
   - System information

---

## 🗺️ Roadmap

- [ ] Mobile app (Android/iOS)
- [ ] Real-time dashboard updates
- [ ] Email notifications
- [ ] SMS alerts for low attendance
- [ ] Biometric authentication
- [ ] Multi-language support
- [ ] Dark mode
- [ ] Advanced analytics
- [ ] Export to Google Sheets
- [ ] Integration with LMS

---

## 📊 System Requirements

### Minimum
- **CPU:** Dual-core 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 2 GB free space
- **OS:** Windows 10, Ubuntu 18.04, macOS 10.14+

### Recommended
- **CPU:** Quad-core 2.5 GHz+
- **RAM:** 8 GB+
- **Storage:** 5 GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional)

---

## 📈 Statistics

- **Recognition Accuracy:** 85-95% (depending on distance)
- **Processing Speed:** 2-5 FPS (CPU) / 15-30 FPS (GPU)
- **Face Detection Range:** 20px to 150px+ face width
- **Supported Students:** Unlimited (tested with 500+)
- **Concurrent Users:** 50+ (with proper hardware)

---

## 🎓 Academic Use

This system was developed as part of an academic project. If you use this system in your research or project, please cite:

```
@software{smart_attendance_system,
  author = {Your Name},
  title = {Smart Attendance System: AI-Powered Face Recognition},
  year = {2025},
  url = {https://github.com/yourusername/smart-attendance-system}
}
```

---

## ⚠️ Disclaimer

This system is designed for **LOCAL USE ONLY** and should be deployed in a controlled environment. For production deployment:

1. Implement HTTPS
2. Use password hashing (bcrypt)
3. Add CSRF protection
4. Implement rate limiting
5. Regular security audits
6. Comply with data privacy regulations (GDPR, etc.)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Git-Suraj-hub/Vision_Mark&type=Date)](https://star-history.com/#Git-Suraj-hub/Vision_Mark&Date)

---

<div align="center">

**Made with ❤️ for educational institutions**


</div>
