import uuid
from fastapi import FastAPI
from dotenv import load_dotenv


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