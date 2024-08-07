from fastapi import Depends, Query
from redis import Redis

from cbir.config import DatabaseSetting, ModelSetting
from cbir.retrieval.indexer import Indexer
from cbir.retrieval.retrieval import ImageRetrieval
from cbir.retrieval.store import Store
from cbir.retrieval.utils import get_redis


def get_store(
    storage_name: str = Query(..., alias="storage"),
    index_name: str = Query(default="index", alias="index"),
    redis: Redis = Depends(get_redis),
) -> Store:
    return Store(storage_name, redis, index_name)


def get_indexer(
    storage_name: str = Query(..., alias="storage"),
    index_name: str = Query(default="index", alias="index"),
    database_settings: DatabaseSetting = Depends(DatabaseSetting.get_settings),
    model_settings: ModelSetting = Depends(ModelSetting.get_settings),
) -> Indexer:
    return Indexer(
        database_settings.data_path,
        storage_name,
        index_name,
        model_settings.n_features,
        model_settings.device.type == "cuda",
    )


def get_retrieval(
    store: Store = Depends(get_store),
    indexer: Indexer = Depends(get_indexer),
) -> ImageRetrieval:
    return ImageRetrieval(store, indexer)
