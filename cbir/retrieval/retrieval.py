"""Image retrieval methods."""

from io import BytesIO
from typing import List
import torch

from cbir.models.model import Model
from cbir.retrieval.database import Database
from cbir.retrieval.indexer import Indexer
from torchvision import transforms
from PIL import Image


class ImageRetrieval:
    def __init__(self, database: Database, indexer: Indexer) -> None:
        """Image retrieval initialisation."""
        self.database = database
        self.indexer = indexer

    def index_image(
        self,
        model: Model,
        image: torch.Tensor,
        storage_name: str,
        index_name: str,
    ) -> List[int]:
        """Index an image."""

        features_extraction = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create a dataset of one image
        inputs = features_extraction(Image.open(BytesIO(image)))
        inputs = torch.unsqueeze(image, dim=0)

        with torch.no_grad():
            outputs = model(inputs.to(model.device)).cpu().numpy()

        last_id = self.database.get_last_id(storage_name, index_name)
        ids = self.indexer.add(last_id, storage_name, index_name, outputs)

        return ids

    def remove_image(
        self,
        name,
        storage_name: str,
        index_name: str,
    ):
        """Remove an image."""
        pass

    def search(
        self,
        model: Model,
        nrt_neigh,
        features,
        storage_name: str,
        index_name: str,
    ):
        """Search for similar images."""
        pass
