"""Image retrieval methods."""

from io import BytesIO
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from cbir.models.model import Model
from cbir.retrieval.indexer import Indexer
from cbir.retrieval.store import Store


class ImageRetrieval:
    def __init__(self, store: Store, indexer: Indexer) -> None:
        """Image retrieval initialisation."""
        self.store = store
        self.indexer = indexer

    def index_image(self, model: Model, image: bytes) -> List[int]:
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
        inputs = features_extraction(Image.open(BytesIO(image)).convert("RGB"))
        inputs = torch.unsqueeze(inputs, dim=0)

        with torch.no_grad():
            outputs = model(inputs.to(model.device)).cpu().numpy()

        last_id = self.store.last()
        ids = self.indexer.add(last_id, outputs)
        self.store.set("last_id", ids[-1] + 1)

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
