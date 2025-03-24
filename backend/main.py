import uuid
from fastapi import FastAPI,  APIRouter, Request
from dotenv import load_dotenv
from RecognitionModel.findmatch import predict_antler
from fastapi.responses import JSONResponse
import requests
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

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
cur = conn.cursor()

class ScanCreate(BaseModel):
    userid: str
    url: str

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

@app.post("/api/match-antler")
async def match_antler_via_url(data: dict):
    userid = data.get("userid")
    file_url = data.get("fileUrl")
    scanid = data.get("scanid") or str(uuid.uuid4())  # optional: use scanid from frontend
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
        "modelUrl": f"https://your-bucket-url/{matchid}.stl"
    }
