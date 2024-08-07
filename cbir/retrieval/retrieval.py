"""Image retrieval methods."""

from io import BytesIO
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from cbir.models.model import Model
from cbir.retrieval.indexer import Indexer
from cbir.retrieval.store import Store


class ImageRetrieval:
    """Image retrieval class."""

    def __init__(self, store: Store, indexer: Indexer) -> None:
        """
        Image retrieval initialisation.

        Args:
            store (Store): The store object.
            indexer (Indexer): The indexer object.
        """
        self.store = store
        self.indexer = indexer

    def index_image(self, model: Model, image: bytes, filename: str) -> List[int]:
        """
        Index an image.

        Args:
            model (Model): The model to extract features.
            image (bytes): The image to be indexed.
            filename (str): The name of the image.

        Returns:
            List[int]: The IDs of the indexed images.
        """

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
        self.store.set(filename, ids[-1])
        self.store.set("last_id", ids[-1] + 1)

        return ids

    def remove_image(self, name: str) -> Optional[int]:
        """
        Remove an image.

        Args:
            name (str): The name of the image to be removed.

        Returns:
            Optional[int]: The ID of the removed image or None if it does not exist.
        """

        label = self.store.get(name)
        if not label:
            return None

        self.indexer.remove(label)
        self.store.remove(name)

        return label

    def search(
        self,
        model: Model,
        nrt_neigh: int,
        features: int,
        storage_name: str,
        index_name: str,
    ) -> None:
        """Search for similar images."""
