import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from pymongo.errors import PyMongoError

load_dotenv("config.env")


class MarketDB:
    def __init__(self):
        # 1. read URI
        self.uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("MONGODB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.client = None
        self.is_connected = False

        try:
            # 2. check uri
            if not self.uri:
                print("❌ CONFIG ERROR: MONGODB_URI is missing in .env")
                return

            # 3. init mongodb client
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                tz_aware=True,
                tlsCAFile=certifi.where()
            )

            # 4. check ping
            self.client.admin.command('hello')

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.is_connected = True
            print(f"MongoDB Initialized: {self.db_name}.{self.collection_name}")

        except Exception as e:
            print(f"Connection Failed: {e}")
            self.is_connected = False

    def upsert_benchmark(self, doc):
        """Updates or inserts the market data."""
        if not self.is_connected: return False
        try:
            result = self.collection.update_one(
                {"role_name": doc["role_name"]},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id:
                print(f"Created new benchmark for: {doc['role_name']}")
            else:
                print(f"Updated existing benchmark for: {doc['role_name']}")
            return True
        except PyMongoError as e:
            print(f"Upsert Failed: {e}")
            return False

    def get_benchmark(self, role_name):
        """Retrieves data for a role."""
        if not self.is_connected: return None
        try:
            return self.collection.find_one({"role_name": role_name})
        except PyMongoError as e:
            print(f"Query Failed: {e}")
            return None

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")