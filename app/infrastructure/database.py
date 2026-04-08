"""
MongoDB connection manager.

Provides a singleton MongoClient with proper lifecycle management.
The client is created once during application startup and closed on shutdown.
"""

from __future__ import annotations

import logging

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages the MongoDB connection lifecycle.

    Usage:
        db_manager = DatabaseManager(uri, db_name)
        db_manager.connect()      # call during startup
        collection = db_manager.get_collection("knowledge_base")
        db_manager.close()        # call during shutdown
    """

    def __init__(self, uri: str, db_name: str) -> None:
        self._uri = uri
        self._db_name = db_name
        self._client: MongoClient | None = None

    def connect(self) -> None:
        """Create the MongoClient and verify connectivity."""
        if self._client is not None:
            logger.warning("DatabaseManager.connect() called but already connected")
            return

        logger.info("Connecting to MongoDB: db=%s", self._db_name)
        self._client = MongoClient(self._uri)

        # Ping to verify the connection is live.
        try:
            self._client.admin.command("ping")
            logger.info("MongoDB connection verified successfully")
        except Exception:
            logger.exception("MongoDB connection ping failed")
            raise

    def close(self) -> None:
        """Close the MongoClient."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")

    @property
    def client(self) -> MongoClient:
        """Return the active MongoClient."""
        if self._client is None:
            raise RuntimeError(
                "DatabaseManager not connected. Call connect() first."
            )
        return self._client

    def get_database(self) -> Database:
        """Return the application database."""
        return self.client[self._db_name]

    def get_collection(self, collection_name: str) -> Collection:
        """Return a collection from the application database."""
        return self.get_database()[collection_name]
