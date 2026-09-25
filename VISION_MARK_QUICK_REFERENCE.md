**Created by: Suraj Singh Mehta**

# Vision Mark — Quick Reference (one page)

> Keep this open while working. Details → `VISION_MARK_DEVELOPER_GUIDE.md`.

## What it is

Face-recognition classroom attendance. **FastAPI + Jinja2 + vanilla JS + MySQL + InsightFace (ArcFace)**.
One Python process does web + AI + webcam. Run everything from `pythonProject4/`.

## Install

```powershell
git clone https://github.com/Git-Suraj-hub/Vision_Mark.git
cd Vision_Mark
python -m venv venv; venv\Scripts\activate
pip install fastapi "uvicorn[standard]" python-multipart jinja2 python-dotenv `
            mysql-connector-python insightface retina-face opencv-python numpy scipy torch onnxruntime
```
> There is **no `requirements.txt`** in this repo (README is wrong).

## Environment

`.env` at repo root (only config that exists):

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=<YOUR_SECRET>
DB_NAME=face_detection
```
> `Far/*.py` never call `load_dotenv()` → they fail until you export these manually.

## Run

```powershell
Get-Service MySQL80                       # must be Running
cd D:\Vision_Mark\pythonProject4
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000        # add --reload for dev
```
First boot downloads `buffalo_l` (288 MB) to `~\.insightface\models\buffalo_l` → ~2–3 min.
**No worker/queue/Redis/Docker/frontend server exists. Use ONE worker only** (in-memory sessions + webcam thread).

## Commands

| Task | Command |
| --- | --- |
| Dev server | `python -m uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000` (in `pythonProject4`) |
| Face engine CLI | `python backend\Recognize2.py` (menu: webcam/image/video/DB-debug) |
| Enroll step 1 | `python backend\student_embeddings.py` → `student_embeddings` (asks to clear old first) |
| Enroll step 2 | `python backend\student_detail_embedding.py` → averages in `student_detail` |
| Capture tool | `python Far\Distance_Categorize.py` |
| Train SVM / KNN | `python Far\Model.py` / `python Far\model1.py` → `models/*.pkl` (unused by app) |
| Build / Test / Lint / Migrate / Docker | **NONE EXIST** — verify manually (below) |

## URLs

```
http://127.0.0.1:8000/                  landing
http://127.0.0.1:8000/login/student     /login/teacher
http://127.0.0.1:8000/student/dashboard  /teacher/dashboard
http://127.0.0.1:8000/docs              interactive API docs
http://127.0.0.1:8000/static/css/style.css
```

## Important folders

```
.env                                  DB credentials (committed to git — rotate if public)
pythonProject4/backend/server.py      16 routes, sessions, SUBJECT_TABLE_MAP, attendance callback
pythonProject4/backend/Recognize2.py  AI: detect → preprocess → embed → 4-level match
pythonProject4/backend/templates/     index, login_student, login_teacher, student, teacher
pythonProject4/backend/static/        css, images, captures/, uploads/
pythonProject4/Dataset/               enrollment photos: <ENROLLMENT>_<Name>/*.jpg
pythonProject4/Far/                   experiments (SVM/KNN, capture tool) — not wired in
Demo/                                 screenshots only
```

## Important APIs

| Method | Endpoint | Auth | Purpose |
| --- | --- | --- | --- |
| POST | `/api/login` | — | `role,username,password` → `session` cookie (6 h) |
| POST | `/api/logout` | cookie | kill session |
| GET | `/api/current_user` | cookie | who am I |
| GET | `/api/current_subject` | **none** | class running now (from `dummytimetable`) |
| GET | `/api/student/attendance/{enrollment_no}` | **none (IDOR)** | all subjects for one student |
| GET | `/api/teacher/attendance` | teacher | own subject records + stats |
| POST | `/api/mark_attendance` | teacher | `enrollment_no,status` (5 validations) |
| POST | `/api/start_webcam` | teacher | spawns recognition thread |
| GET | `/api/stop_webcam` | teacher | stops it |
| GET | `/api/recent_recognitions` | teacher | last 20 (polled every 3 s) |
| POST | `/api/test_image` | teacher | multipart `file` → annotated result image |

## AI pipeline

```
frame (every 2nd) → SCRFD/det_10g 640x640 (det>0.2)  → if width≥150px use RetinaFace(0.5)
→ crop → CLAHE + denoise + sharpen → ArcFace w600k_r50 → 512-d, L2-normalized
→ bucket by width: ≥150 very_close, ≥80 close, ≥40 mid, ≥20 far, else very_far
→ cosine match (dot): L1 same bucket → L2 adjacent bucket → L3 bucket average → L4 global average
→ thresholds: 0.25 / 0.25 / 0.30 / 0.35 / 0.40  → INSERT attendance (recognized_by='auto_face')
```

## Database (MySQL `face_detection`, 16 tables)

```
students(4)  teachers(10)  student_detail(2)  student_embeddings(68)  dummytimetable(168)
attendance tables: toc_attendance, se_attendence, dbms_attendence, dbmslab_attendence,
                   selab_attendance, personality_attendance, ics_attendance,
                   ai_mark_attendance, minorproject_attendance      (FK → student_detail)
dead tables: attendance, timetable
```

```powershell
python -c "import mysql.connector as m;c=m.connect(host='localhost',user='root',password='<YOUR_SECRET>',database='face_detection');cur=c.cursor();cur.execute('SHOW TABLES');[print(r[0]) for r in cur.fetchall()]"
# backup: mysqldump -u root -p face_detection > face_detection.sql   (CLI not on PATH here)
```

## Verify a running server

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing).StatusCode      # 200
Invoke-WebRequest http://127.0.0.1:8000/api/current_subject -UseBasicParsing
try { Invoke-WebRequest http://127.0.0.1:8000/api/teacher/attendance } catch { $_.Exception.Response.StatusCode.value__ }  # 401
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'backend'` | `cd pythonProject4` first |
| Startup hangs | 288 MB model download — wait / check internet |
| DB `Can't open named pipe` / access denied | `.env` values, `Get-Service MySQL80` |
| `Far\*.py` can't connect | they lack `load_dotenv()` |
| "Invalid subject" when marking | `teachers.subject` ≠ `SUBJECT_TABLE_MAP` key (3 teachers affected) |
| "No class running" | add a `dummytimetable` row for today's slot |
| FK error on insert | enrollment missing in `student_detail` |
| Webcam won't open | camera index `0` in `Recognize2.py:507`, Windows camera privacy |
| Always "Unknown" | run `Recognize2.py` option 4; regenerate embeddings; lighting |
| New embeddings not used | embeddings load at startup → restart |

## Git

```powershell
git status
git checkout -b feature/<name>
git add <files you touched>          # Dataset/ and static/ are huge binaries
git commit -m "feature: what changed"
git push -u origin feature/<name>
```
Single branch `main`, 9 commits (all Dec 2025), no CI, no tests, no conventions.

## Top gotchas

1. README documents features that don't exist (reports, absent marking, `requirements.txt`, `database/*.sql`, sample logins).
2. `.env` with the real DB password is committed; passwords in DB are plaintext.
3. Single worker only; camera + `cv2.imshow` run on the **server** machine, not in the browser.
4. `SUBJECT_TABLE_MAP` must be edited in **two** places: `server.py:68` and `Recognize2.py:671`.
5. `student_embeddings.py` has a hardcoded dataset path from the old machine — edit line 28.
