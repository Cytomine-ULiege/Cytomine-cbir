#  Copyright 2023 Cytomine ULiège
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Database communications"""

from typing import List, Tuple

import torch
from redis import Redis  # type: ignore

from cbir.config import DatabaseSetting
from cbir.models.model import Model


class Database:
    """Database to store the indices."""

    def __init__(self, settings: DatabaseSetting) -> None:
        """Database initialisation."""
        self.settings = settings
        self.redis = Redis(host=settings.host, port=settings.port, db=settings.db)

    def get_last_id(self, storage_name: str, index_name: str) -> str:
        return self.redis.get(f"{storage_name}:{index_name}:last_id") or "0"

    def update_last_id(self, storage_name: str, index_name: str, last_id):
        self.redis.set(f"{storage_name}:{index_name}:last_id", last_id)

    def contains(self, storage_name: str, index_name: str, name: str) -> bool:
        """Check if a filename is in the index database."""
        return self.redis.get(f"{storage_name}:{index_name}:{name}") is not None

    def save(
        self,
        storage_name: str,
        index_name: str,
        name: str,
        ids: List[int],
    ) -> None:
        """Save the index database."""
        for idx in ids:
            self.redis.set(f"{storage_name}:{index_name}:{name}", idx)

    def remove(self, storage_name: str, index_name: str, name: str) -> None:
        """Remove an image from the index database."""

        self.redis.delete(f"{storage_name}:{index_name}:{name}")

    def search_similar_images(
        self,
        model: Model,
        query: torch.Tensor,
        nrt_neigh: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Search similar images given a query image."""

        inputs = torch.unsqueeze(query, dim=0)

        with torch.no_grad():
            outputs = model(inputs).cpu().numpy()

        distances, labels = self.index.search(outputs, nrt_neigh)
        distances, labels = distances.squeeze().tolist(), labels.squeeze().tolist()

        if nrt_neigh == 1:
            distances = [distances]
            labels = [labels]

        # Return only valid results
        stop = labels.index(-1) if -1 in labels else len(labels)
        filenames = [self.redis.get(str(l)).decode("utf-8") for l in labels[:stop]]

        return filenames, distances[:stop]
