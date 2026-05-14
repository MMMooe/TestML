from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.routers import jobs, uploads
from app.runtime import assert_runtime_ready, get_runtime_info
from app.storage import Storage

settings = get_settings()

app = FastAPI(title="Model Evaluation Gallery API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(uploads.router)
app.include_router(jobs.router)


@app.on_event("startup")
def startup() -> None:
    Storage(settings)
    assert_runtime_ready()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": settings.app_mode}


@app.get("/runtime")
async def runtime():
    return get_runtime_info()


@app.get("/assets/{asset_path:path}")
async def assets(asset_path: str) -> FileResponse:
    storage = Storage(settings)
    try:
        path = storage.resolve_asset(asset_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid asset path") from error
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)