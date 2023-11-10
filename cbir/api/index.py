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

"""Image indexing"""

from fastapi import APIRouter, UploadFile, File

from pydantic import BaseModel
from cbir_tfe.models import Model
from cbir_tfe.db import Database
from cbir.config import ModelSetting

config = ModelSetting.get_settings()
router = APIRouter()


class IndexBody(BaseModel):
    db_name: str = "db"
    extractor: str
    rewrite: bool = False


@router.post("/images/index")
async def index_image(image: UploadFile = File(...)):
    """Index the given image."""
    print(image)

    model = Model(
        model=config.extractor,
        use_dr=config.use_dr,
        num_features=config.n_features,
        name=config.weights,
        device="cuda:0",
    )

    database = Database('db', model, load=config.load_database)
    #database.add_dataset("TODO", config.extractor, config.generalise, label = True)

@router.post('/file')
def _file_upload(
        my_file: UploadFile = File(...),
):
    print(my_file)
    return {
        "name": my_file.filename,
    }
