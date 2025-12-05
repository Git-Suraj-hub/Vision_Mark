import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)
count = {'very_close': 0, 'close': 0, 'mid': 0, 'far': 0}

while True:
    ret, frame = cap.read()
    faces = app.get(frame)
    
    if faces:
        bbox = faces[0].bbox.astype(int)
        width = bbox[2] - bbox[0]
        
        # Show category in REAL-TIME
        if width >= 150:
            category = "very_close"
            color = (0,0,255)  # Red
        elif width >= 80:
            category = "close"
            color = (0,255,255)  # Yellow
        elif width >= 40:
            category = "mid"
            color = (0,255,0)  # Green
        else:
            category = "far"
            color = (255,0,0)  # Blue
        
        # Draw box + category
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
        cv2.putText(frame, f"{category} ({width}px) - Press SPACE", 
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Press SPACE to save
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            cv2.imwrite(f"photo_{category}_{count[category]}.jpg", frame)
            count[category] += 1
            print(f"✓ Saved {category}: {count[category]}")
    
    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()