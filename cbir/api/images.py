"""Image API"""

import json
from io import BytesIO
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from cbir.api.utils.utils import get_retrieval
from cbir.config import DatabaseSetting
from cbir.retrieval.retrieval import ImageRetrieval

router = APIRouter()


@router.post("/images")
async def index_image(
    request: Request,
    image: UploadFile,
    storage_name: str = Query(..., alias="storage"),
    index_name: str = Query(default="index", alias="index"),
    retrieval: ImageRetrieval = Depends(get_retrieval),
    settings: DatabaseSetting = Depends(DatabaseSetting.get_settings),
) -> JSONResponse:
    """
    Index the given image into the specified storage and index.

    Args:
        request (Request): The incoming HTTP request.
        image (UploadFile): The image file to be indexed.
        storage_name (str): The name of the storage where the index is stored.
        index_name (str): The name of the index where the image features will be added.
        retrieval (ImageRetrieval): The image retrieval object.
        settings (DatabaseSetting): The database settings.

    Returns:
        JSONResponse: A JSON response containing the ID of the newly indexed image.
    """

    if image.filename is None:
        raise HTTPException(status_code=404, detail="Image filename not found!")

    if retrieval.store.contains(image.filename):
        raise HTTPException(status_code=409, detail="Image filename already exist!")

    if not storage_name:
        raise HTTPException(status_code=404, detail="Storage is required")

    base_path = Path(settings.data_path)
    storage_path = base_path / storage_name
    if not storage_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Storage '{storage_name}' not found.",
        )

    content = await image.read()

    model = request.app.state.model

    ids = retrieval.index_image(model, content, image.filename)

    return JSONResponse(
        content={
            "ids": ids,
            "storage": storage_name,
            "index": index_name,
        }
    )


@router.delete("/images/{filename}")
def remove_image(
    filename: str,
    storage_name: str = Query(..., alias="storage"),
    index_name: str = Query(default="index", alias="index"),
    retrieval: ImageRetrieval = Depends(get_retrieval),
) -> JSONResponse:
    """
    Remove an indexed image.

    Args:
        filename (str): The name of the image to be removed.
        storage_name (str): The name of the storage where the index is stored.
        index_name (str): The name of the index where the image features will be added.
        retrieval (ImageRetrieval): The image retrieval object.

    Returns:
        JSONResponse: A JSON response containing the ID of the deleted image.
    """

    if not retrieval.store.contains(filename):
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    label = retrieval.remove_image(filename)

    return JSONResponse(
        content={
            "id": label,
            "storage": storage_name,
            "index": index_name,
        }
    )


@router.post("/images/retrieve")
async def retrieve_image(
    request: Request,
    nrt_neigh: int = Form(),
    image: UploadFile = File(),
) -> Response:
    """Retrieve similar images from the database."""

    database = request.app.state.database
    model = request.app.state.model

    content = await image.read()
    features_extraction = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    filenames, distances = database.search_similar_images(
        model,
        features_extraction(Image.open(BytesIO(content))),
        nrt_neigh=nrt_neigh,
    )

    return Response(
        content=json.dumps({"filenames": filenames, "distances": distances}),
    )
