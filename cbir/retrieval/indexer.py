"""Indexer class for indexing images and their features."""

import os
from typing import List, Tuple

import faiss
import numpy as np
import torch


class Indexer:
    """Indexer class for indexing images and their features."""

    def __init__(self, data_path: str, n_features: int, gpu: bool = False) -> None:
        """
        Indexer initialisation.

        Args:
            data_path (str): Path to the base storage.
            n_features (int): Number of features in the index.
            gpu (bool): Whether to use GPU for indexing or not.
        """
        self.data_path = data_path
        self.n_features = n_features

        if gpu:
            self.resources = faiss.StandardGpuResources()

    def index(self, storage_name: str, index_name: str) -> faiss.Index:
        """
        Get the Faiss index for the given storage.

        Args:
            storage_name (str): The name of the storage.
            index_name (str): The name of the index.

        Returns:
            faiss.Index: The Faiss index.
        """

        index_path = os.path.join(self.data_path, storage_name, index_name)

        if os.path.isfile(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatL2(self.n_features)
            index = faiss.IndexIDMap(index)

        if self.gpu:
            return faiss.index_cpu_to_gpu(self.resources, 0, index)

        return index

    def save(self, storage_name: str, index_name: str) -> None:
        """
        Save the index to the file.

        Args:
            index (faiss.Index): The Faiss index.
        """

        index = self.index(storage_name, index_name)
        index = faiss.index_gpu_to_cpu(self.index) if self.gpu else self.index
        faiss.write_index(index, self.data_path)

    def add(
        self,
        last_id: int,
        storage_name: str,
        index_name: str,
        images: torch.Tensor,
    ) -> List[int]:
        """
        Index the given images in the provided index.

        Args:
            last_id (int): The last ID in the index.
            images (torch.Tensor): The images to be indexed.

        Returns:
            List[int]: A list of IDs of the indexed images.
        """

        ids = np.arange(last_id, last_id + images.shape[0])

        index = self.index(storage_name, index_name)
        index.add_with_ids(images, ids)

        self.save(storage_name, index_name)

        return ids.tolist()

    def remove(self, storage_name: str, index_name: str, id: int) -> None:
        """
        Remove an image from the given index.

        Args:
            index (faiss.Index): The Faiss index.
            id (int): The ID of the image to be removed.
        """

        id_selector = faiss.IDSelectorRange(id, id + 1)

        index = self.index(storage_name, index_name)
        index = faiss.index_gpu_to_cpu(index) if self.gpu else index
        index.remove_ids(id_selector)

        self.save(storage_name, index_name)

    def search(self, image: np.array, nrt_neigh: int) -> Tuple[List[str], List[float]]:
        """
        Search similar images given a query image.

        Args:
            image (np.array): The query image.
            nrt_neigh (int): The number of nearest neighbours to search.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing the list of image IDs and their distances
        """

        distances, labels = self.index.search(image, nrt_neigh)
        distances, labels = distances.squeeze().tolist(), labels.squeeze().tolist()

        if nrt_neigh == 1:
            distances = [distances]
            labels = [labels]

        # Return only valid results
        stop = labels.index(-1) if -1 in labels else len(labels)

        return labels[:stop], distances[:stop]
