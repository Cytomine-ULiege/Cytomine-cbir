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

"""Environment parameters"""

from pydantic_settings import BaseSettings


class DatabaseSetting(BaseSettings):
    """Database settings."""

    filename: str = "db"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    image_path: str = "/tmp/images/"

    @staticmethod
    def get_settings():
        """Get the settings.

        Returns:
            DatabaseSetting: The database settings.
        """
        return DatabaseSetting(_env_file="database.env", _env_file_encoding="utf-8")


class ModelSetting(BaseSettings):
    """Model settings."""

    device: str = "cpu"
    extractor: str = "resnet"
    generalise: int = 0
    n_features: int = 128
    use_dr: bool = False
    weights: str = f"/weights/{extractor}"

    @staticmethod
    def get_settings():
        """Get the settings.

        Returns:
            ModelSetting: The model settings.
        """
        return ModelSetting(_env_file=".env", _env_file_encoding="utf-8")