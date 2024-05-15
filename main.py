import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
from ultralytics import YOLO
import uvicorn

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/infusion-drop.pt")

def detect_drops(frame):
    results = model.predict(frame)[0]
    detections = []
    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        confidence = result.conf[0]
        class_id = result.cls[0]
        detections.append({
            'x1': int(x1), 'y1': int(y1),
            'x2': int(x2), 'y2': int(y2),
            'confidence': float(confidence),
            'class_id': int(class_id),
            'class_name': model.names[int(class_id)]
        })
    return detections

def count_total_drops(frame):
    detections = detect_drops(frame)
    return len(detections)

def process_video(video_path):
    total_drops = 0
    drops_in_one_minute = 0
    start_time = time.time()

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1 / fps

    for frame_idx in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        drop_count = count_total_drops(frame)
        total_drops += drop_count

        current_time = time.time()
        time_diff = current_time - start_time
        if time_diff >= 60:
            drops_in_one_minute = total_drops - drops_in_one_minute
            start_time = current_time

        time.sleep(frame_interval)

    video.release()
    cv2.destroyAllWindows()

    return {"total_drops": total_drops, "drops_in_one_minute": drops_in_one_minute}

@app.get("/")
async def check_health():
    return "Healthy"

@app.post("/detect_objects")
async def detect_objects(file: UploadFile = File(...)):
    allowed_extensions = {'.mp4', '.mkv'}
    if file.filename.endswith(tuple(allowed_extensions)):
        try:
            video_path = f"uploads/{file.filename}"
            with open(video_path, "wb") as buffer:
                buffer.write(file.file.read())

            result = process_video(video_path)
            os.remove(video_path)
            return JSONResponse(content=result)

        except Exception as e:
            return JSONResponse(content={"error": str(e)})

    else:
        return JSONResponse(content={"error": "Invalid file format. Only .mp4 and .mkv files are allowed."})

if __name__ == "__main__":
    uvicorn.run(app)