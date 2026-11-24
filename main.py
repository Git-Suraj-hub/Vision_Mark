"""
Vision Mark - Academic Facial Recognition Attendance Platform
Root application entrypoint for cloud deployment (Render, Railway, HuggingFace, Docker, VPS)
"""

import sys
import os

# Ensure pythonProject4 is in system path
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pythonProject4")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from backend.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
