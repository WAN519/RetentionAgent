import os
import json
import datetime
import requests
from dotenv import load_dotenv


# Import your existing marketDB class from your db file
# from your_db_file import marketDB

class MarketDataCoordinator:
    def __init__(self, db_handler):
        load_dotenv()
        self.api_key = os.getenv('BLS_API_KEY')
        self.base_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
        self.db = db_handler  # This is your marketDB instance

    def _post_request(self, payload):
        headers = {'Content-type': 'application/json'}
        response = requests.post(self.base_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_latest_market_data(self, series_id, role_name):
        """
        Main entry point:
        1. Checks MongoDB for existing data.
        2. If data is fresh (<30 days), returns it.
        3. If stale or missing, syncs with BLS and updates MongoDB.
        """
        # 1. Check MongoDB Cache
        cached_doc = self.db.collection.find_one({"role_name": role_name})

        if cached_doc and 'sync_timestamp' in cached_doc:
            last_sync = cached_doc['sync_timestamp']
            # Check if cache is less than 30 days old
            if (datetime.datetime.utcnow() - last_sync).days < 30:
                print(f"✅ [CACHE HIT] Serving {role_name} from MongoDB (Synced on {last_sync.date()})")
                return cached_doc

        # 2. Cache Miss: Run Full Sync
        print(f"🔄 [CACHE MISS/EXPIRED] Syncing {role_name} with BLS API...")
        fresh_data = self._run_bls_sync_logic(series_id, role_name)

        if fresh_data:
            # Add native UTC datetime for MongoDB indexing
            fresh_data['sync_timestamp'] = datetime.datetime.utcnow()

            # 3. Upsert to MongoDB (Update if exists, Insert if not)
            self.db.collection.update_one(
                {"role_name": role_name},
                {"$set": fresh_data},
                upsert=True
            )
            print(f"💾 [MONGODB UPDATED] Benchmark for {role_name} is now current.")
            return fresh_data

        return None

    def _run_bls_sync_logic(self, series_id, role_name):
        """Internal logic for salary/CPI alignment."""
        current_year = datetime.datetime.now().year
        anchor_year = None
        base_hourly = 0.0

        # Step 1: Find Latest Salary Year
        for year in range(current_year - 1, current_year - 5, -1):
            payload = {
                "seriesid": [series_id],
                "startyear": str(year), "endyear": str(year),
                "registrationkey": self.api_key
            }
            res = self._post_request(payload)
            points = res.get('Results', {}).get('series', [{}])[0].get('data', [])
            if points:
                anchor_year = str(year)
                base_hourly = float(points[0]['value'])
                break

        if not anchor_year: return None

        # Step 2: Align CPIs
        cpi_id = "CUUR0000SA0"
        # Fetch Anchor Year CPI (Baseline)
        base_cpi_res = self._post_request(
            {"seriesid": [cpi_id], "startyear": anchor_year, "endyear": anchor_year, "registrationkey": self.api_key})
        base_cpi = float(base_cpi_res['Results']['series'][0]['data'][0]['value'])

        # Fetch Current 2026 CPI (Target)
        now_cpi_res = self._post_request(
            {"seriesid": [cpi_id], "startyear": str(current_year - 1), "endyear": str(current_year),
             "registrationkey": self.api_key})
        current_cpi_node = now_cpi_res['Results']['series'][0]['data'][0]
        current_cpi = float(current_cpi_node['value'])

        # Step 3: Calculation
        annual_base = base_hourly * 2080
        inflation_factor = current_cpi / base_cpi
        predicted = annual_base * inflation_factor * 1.03  # 3% Tech Premium

        return {
            "role_name": role_name,
            "series_id": series_id,
            "data": {
                "anchor_year": anchor_year,
                "base_annual": round(annual_base, 2),
                "current_cpi": current_cpi,
                "inflation_rate": f"{((inflation_factor - 1) * 100):.2f}%"
            },
            "predicted_market_salary": round(predicted, 2)
        }