import uuid
from fastapi import FastAPI

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "postgresql://postgres:andrew@localhost:5432/buck3d"

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
