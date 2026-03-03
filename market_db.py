import os
import sys
import datetime
import traceback
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from bson import ObjectId

MONGO_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("MONGODB_NAME")
COLLECTION_NAME = "market_benchmarks" # Dedicated collection for salary data

class MarketDB:
    def __init__(self):
        self.client = None
        self.is_connected = False
        try:
            if not MONGO_URI:
                raise ValueError("MONGODB_URI is not set in environment variables")

            self.client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                tlsAllowInvalidCertificates=True
            )

            self.client.list_database_names()

            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.is_connected = True
            print(f"✓ Connected to MongoDB: {DB_NAME}")
        except Exception as e:
            print(f"✗ Connection Failed: {e}")

    def upsert_market_benchmark(self, market_doc):
        """Updates or Inserts salary data based on role_name."""
        if not self.is_connected: return False
        try:
            query = {"role_name": market_doc["role_name"]}
            self.collection.update_one(query, {"$set": market_doc}, upsert=True)
            return True
        except Exception as e:
            print(f"✗ Upsert failed: {e}")
            return False

    def get_benchmark_by_role(self, role_name):
        """Retrieves cached data for a specific role."""
        if not self.is_connected: return None
        return self.collection.find_one({"role_name": role_name})

    def close(self):
        if self.client: self.client.close()