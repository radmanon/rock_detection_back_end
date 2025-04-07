from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import cv2
import time
import threading

from predict import predict_on_image
from webcam_detect import run_frame  # For live stream endpoint


# -------------------- FastAPI Setup --------------------
app = FastAPI()

# Serve predicted images from 'results/' folder
app.mount("/results", StaticFiles(directory="results"), name="results")

# CORS setup (for frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Prediction Endpoint --------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    upload_path = f"uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    result_path = predict_on_image(upload_path)
    filename = os.path.basename(result_path)
    return JSONResponse(content={
        "status": "✅ Prediction complete",
        "result_path": f"/results/{filename}"
    })

# -------------------- Live Detection via Separate Window --------------------
@app.get("/start-detection")
def start_live_detection():
    def run_live():
        import webcam_detect
        webcam_detect.run_camera()  # Opens cv2 GUI locally
    thread = threading.Thread(target=run_live)
    thread.start()
    return {"status": "🎥 Live detection started"}

# -------------------- Video Feed Stream (for embedded webcam) --------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = run_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


from fastapi.responses import FileResponse

@app.get("/live.html")
def serve_live_html():
    return FileResponse("live.html")


# -------------------- Start the Server --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
