"""Storage API"""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/storages")
async def get_storages(request: Request) -> JSONResponse:
    """Get all storages."""

    settings = request.app.state.database.settings
    base_path = Path(settings.path)

    if not base_path.is_dir():
        return HTTPException(
            status_code=404,
            detail="Base path not found or is not a directory",
        )

    folder_names = [f.name for f in base_path.iterdir() if f.is_dir()]

    return JSONResponse(content=folder_names)


@router.post("/storages")
async def create_storage(request: Request, name: str) -> JSONResponse:
    """Create a new storage."""

    settings = request.app.state.database.settings
    base_path = Path(settings.path)
    storage_path = base_path / name

    if storage_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Storage with name '{name}' already exists.",
        )

    try:
        storage_path.mkdir()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create storage: {str(e)}",
        ) from e

    return JSONResponse(content=f"Created storage with name: {name}")


@router.get("/storages/{name}")
async def get_storage(request: Request, name: str) -> JSONResponse:
    """Get a specific storage."""

    settings = request.app.state.database.settings
    base_path = Path(settings.path)
    storage_path = base_path / name

    if not storage_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Storage with name '{name}' not found.",
        )

    return JSONResponse(content={"name": name}, media_type="application/json")


@router.delete("/storages/{name}")
async def delete_storage(request: Request, name: str) -> JSONResponse:
    """Delete a specific storage and its content."""

    settings = request.app.state.database.settings
    base_path = Path(settings.path)
    storage_path = base_path / name

    if not storage_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Storage with name '{name}' not found.",
        )

    try:
        shutil.rmtree(storage_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete storage: {str(e)}",
        ) from e

    return JSONResponse(content=f"Deleted storage with name: {name}")
