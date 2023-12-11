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

import faiss
from redis import Redis


class Database:
    """Database to store the indices."""

    def __init__(
        self,
        filename: str,
        n_features: int,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        load: bool = False,
        gpu: bool = False,
    ) -> None:
        """Database initialisation."""
        self.filename = filename
        self.redis = Redis(host=host, port=port, db=db)

        if load:
            self.index = faiss.read_index(filename)
        else:
            self.index = faiss.IndexFlatL2(n_features)
            self.index = faiss.IndexIDMap(self.index)

            self.redis.flushdb()
            self.redis.set("last_id", 0)

        if gpu:
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, 0, self.index)

    def save(self):
        """Save the index to the file."""

        faiss.write_index(self.index, self.filename)
