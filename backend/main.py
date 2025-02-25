from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "test test"}

from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = FastAPI()

DATABASE_URL = "postgresql://postgres:andrew@localhost:5432/buck3d"

conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
cur = conn.cursor()

class ScanCreate(BaseModel):
    userid: str
    url: str

@app.post("/api/scans")
def create_scan(scan: ScanCreate):
    cur.execute(
        "INSERT INTO 2dscans (userid, url) VALUES (%s, %s) RETURNING *",
        (scan.userid, scan.url),
    )
    conn.commit()
    return {"message": "Scan saved successfully"}
