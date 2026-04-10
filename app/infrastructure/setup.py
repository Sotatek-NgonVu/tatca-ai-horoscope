"""
Database setup utilities.

Separates MongoDB collection and index creation from the application
lifespan in ``main.py``, keeping ``lifespan`` a clean orchestrator.
"""

from __future__ import annotations

import logging

from pymongo.database import Database

from app.config.settings import Settings

logger = logging.getLogger(__name__)


def ensure_collections_and_indexes(db: Database, settings: Settings) -> None:
    """
    Idempotently create required collections and indexes.

    Safe to call on every startup — ``create_collection`` is a no-op
    when the collection already exists.
    """
    existing = set(db.list_collection_names())

    required_collections = [
        settings.MONGODB_CHAT_COLLECTION,
        settings.MONGODB_USERS_COLLECTION,
        settings.MONGODB_COLLECTION_NAME,
    ]
    for col_name in required_collections:
        if col_name not in existing:
            db.create_collection(col_name)
            logger.info("Created collection: %s", col_name)

    # Compound index on chat_histories for efficient per-user history queries
    chat_col = db[settings.MONGODB_CHAT_COLLECTION]
    chat_col.create_index([("user_id", 1), ("created_at", -1)])
    logger.info(
        "Ensured index on %s(user_id, created_at)",
        settings.MONGODB_CHAT_COLLECTION,
    )

    # Unique index on users collection
    users_col = db[settings.MONGODB_USERS_COLLECTION]
    users_col.create_index("user_id", unique=True)
    logger.info(
        "Ensured unique index on %s(user_id)",
        settings.MONGODB_USERS_COLLECTION,
    )
