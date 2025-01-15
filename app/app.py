import os
from fastapi import FastAPI, HTTPException, Body

from .whisper import process_audio_file

app = FastAPI()

FILES_DIR = os.environ.get("FILES_DIR", "/tmp")


@app.post("/")
def root(
    file_name: str = Body(),
    batch_size: int = Body(default=32),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
):
    file_path = os.path.join(FILES_DIR, file_name)
    try:
        return process_audio_file(
            file_path,
            batch_size,
            timestamp,
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
