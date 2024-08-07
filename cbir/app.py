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

"""Content Based Image Retrieval API"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from cbir import __version__
from cbir.api import images, searches, storages
from cbir.config import Settings, get_settings
from cbir.models.model import Model
from cbir.models.resnet import Resnet


def load_model(settings: Settings) -> Model:
    """Load the weights of the model."""

    state = torch.load(settings.weights, map_location=settings.device)

    model = Resnet(n_features=settings.n_features, device=settings.device)
    model.load_state_dict(state)
    model.to(settings.device)

    return model


@asynccontextmanager
async def lifespan(local_app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan of the app."""

    # Initialisation
    local_app.state.model = load_model(get_settings())

    yield


app = FastAPI(
    title="Cytomine Content Based Image Retrieval Server",
    description="Cytomine Content Based Image Retrieval Server (CBIR) HTTP API.",
    version=__version__,
    lifespan=lifespan,
    license_info={
        "name": "Apache 2.0",
        "identifier": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)
app.include_router(router=images.router, prefix="/api")
app.include_router(router=searches.router, prefix="/api")
app.include_router(router=storages.router, prefix="/api")
