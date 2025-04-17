import uuid
from fastapi import FastAPI,  APIRouter, Request, File, UploadFile, Form
from dotenv import load_dotenv
from RecognitionModel.findmatch import predict_antler
from fastapi.responses import JSONResponse
import requests
import os
from PIL import Image
from io import BytesIO
from ScoringModel.demo import detect_deer_parts, draw_distances, load_model
from fastapi.staticfiles import StaticFiles


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "up and running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp", exist_ok=True)

# Mount the static file server
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
cur = conn.cursor()

class ScanCreate(BaseModel):
    userid: str
    url: str

class RenameScan(BaseModel):
    name: str

@app.post("/api/scans")
def create_scan(scan: ScanCreate):
    scanid = str(uuid.uuid4())
    cur.execute(
        'INSERT INTO "Scan2D" (scanid, userid, url) VALUES (%s ,%s, %s) RETURNING *',
        (scanid, scan.userid, scan.url),
    )
    conn.commit()
    return {"message": "Scan saved successfully"}

@app.get("/api/scans")
def get_scans(userid : str):
    cur.execute('SELECT * FROM "Scan2D" WHERE userid = %s ORDER BY "createdAt" DESC',
                (userid,))
    scans = cur.fetchall()
    return scans

@app.get("/api/matches")
def get_matches(userid: str):
    cur.execute(
        'SELECT * FROM "Scan2DMatch" WHERE scanid IN (SELECT scanid FROM "Scan2D" WHERE userid = %s)',
        (userid,)
    )
    matches = cur.fetchall()
    return matches

@app.patch("/api/scans")
def rename_scan(userid: str, scanid: str, data: RenameScan):
    try:
        cur.execute(
            'UPDATE "Scan2D" SET "name" = %s WHERE "scanid" = %s AND "userid" = %s RETURNING "scanid", "name";',
            (data.name, scanid, userid) 
        )
        updated_scan = cur.fetchone()
        conn.commit()

        if updated_scan is None:
            raise HTTPException(status_code=404, detail="Scan not found or unauthorized")

        return {"message": "Scan renamed successfully", "scan": updated_scan}
    except Exception as e:
        print("Rename error:", e)
        raise HTTPException(status_code=500, detail=f"Failed to rename scan: {str(e)}")

@app.delete("/api/scans")
def delete_scan(userid: str, scanid: str):
    try:
        cur.execute(
            'DELETE FROM "Scan2D" WHERE "scanid" = %s AND "userid" = %s;',
            (scanid, userid)
        )
        conn.commit()
        return {"message": "Scan deleted successfully"}
    except Exception as e:
        print("Error deleting scan:", e)
        return JSONResponse(status_code=500, content={"message": "Failed to delete scan"})

@app.post("/api/match-antler")
async def match_antler_via_url(data: dict):
    userid = data.get("userid")
    file_url = data.get("fileUrl")
    scanid = data.get("scanid") or str(uuid.uuid4())  
    if not userid or not file_url:
        return JSONResponse(status_code=400, content={"error": "Missing userid or fileUrl"})
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", f"{scanid}.jpg")
    try:
        img_data = requests.get(file_url).content
        with open(temp_path, "wb") as f:
            f.write(img_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Image download failed: {str(e)}"})

    try:
        result = predict_antler(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

    os.remove(temp_path)
    matchid = result.split(" ")[2]  
    match_uuid = str(uuid.uuid4())
    cur.execute(
        'INSERT INTO "Scan2DMatch" (id, scanid, matchid) VALUES (%s, %s, %s)',
        (match_uuid, scanid, matchid)
    )
    conn.commit()

    return {
        "scanid": scanid,
        "match": matchid,
        "modelUrl": f"https://buckview3d.s3.us-east-1.amazonaws.com/3dmodels/{matchid}.stl"
    }
    
@app.post("/api/analyze-parts")
async def analyze_deer_parts(
    file: UploadFile = File(...),
    threshold: float = Form(0.3),
    scale_factor: float = Form(1.0)
):
    """
    Analyze deer parts in an image and calculate distances between them
    """
    # temp directory for storing result file
    os.makedirs("temp", exist_ok=True)
    unique_id = str(uuid.uuid4())
    temp_path = os.path.join("temp", f"{unique_id}.jpg")
    output_path = os.path.join("temp", f"{unique_id}_result.jpg")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        model, class_names = load_model()
        image = Image.open(temp_path).convert("RGB")
        _, detections = detect_deer_parts(
            model, image, class_names, score_threshold=threshold
        )
        result_img, measurements = draw_distances(
            image, detections, output_path=output_path,
            scale_factor=scale_factor, min_score=threshold
        )
        
        # measurments to json for front end
        serializable_measurements = []
        for m in measurements:
            serializable_measurements.append({
                'from': m['from'],
                'to': m['to'],
                'distance': round(float(m['distance']), 2),
                'from_center': [int(c) for c in m['from_center']],
                'to_center': [int(c) for c in m['to_center']]
            })
        class_counts = {}
        for det in detections:
            if det['score'] >= threshold:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
        summary_text = "Deer Parts Analysis Summary:\n"
        for cls, count in class_counts.items():
            summary_text += f"{cls}: {count}, "
        summary_text = summary_text.rstrip(", ") + "\n\n"
        
        summary_text += "Distance Measurements:\n"
        sorted_measurements = sorted(serializable_measurements, key=lambda x: x['distance'])
        for m in sorted_measurements:
            summary_text += f"{m['from']} to {m['to']}: {m['distance']} pixels\n"
        os.remove(temp_path)
        return {
            "detections": [
                {
                    "class": d["class"],
                    "score": round(float(d["score"]), 3),
                    "box": d["box"]
                } for d in detections if d["score"] >= threshold
            ],
            "measurements": serializable_measurements,
            "summary": summary_text,
            "class_counts": class_counts,
            "result_url": f"http://localhost:8000/temp/{os.path.basename(output_path)}"
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error: {e}\n{traceback_str}")
        
        return JSONResponse(
            status_code=500, 
            content={"error": f"Analysis failed: {str(e)}"}
        )