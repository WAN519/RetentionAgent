import datetime
import requests


class MarketDataCoordinator:
    def __init__(self, db_handler, api_key):
        self.db = db_handler
        self.api_key = api_key
        self.url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    def get_market_intelligence(self, series_id, role_name):
        # 1. check cache：if sync in 30 days，return data from mongodb
        cached = self.db.get_benchmark(role_name)
        if cached and 'sync_timestamp' in cached:
            if (datetime.datetime.utcnow() - cached['sync_timestamp']).days < 30:
                print(f"Cache Hit: {role_name}")
                return cached

        # 2. cache failed，run sync
        print(f"Syncing {role_name} from BLS...")
        data = self._fetch_and_calculate(series_id, role_name)

        if data:
            data['sync_timestamp'] = datetime.datetime.utcnow()
            self.db.upsert_benchmark(data)
            return data
        return None

    def _fetch_and_calculate(self, series_id, role_name):
        curr_year = datetime.datetime.now().year
        anchor_year, base_h = None, 0.0

        # get latest year
        for y in range(curr_year - 1, curr_year - 5, -1):
            res = requests.post(self.url, json={
                "seriesid": [series_id], "startyear": str(y), "endyear": str(y), "registrationkey": self.api_key
            }).json()
            pts = res.get('Results', {}).get('series', [{}])[0].get('data', [])
            if pts:
                anchor_year, base_h = str(y), float(pts[0]['value'])
                break

        if not anchor_year: return None

        # get CPI benchmark (same with Salary)
        cpi_id = "CUUR0000SA0"
        b_res = requests.post(self.url, json={"seriesid": [cpi_id], "startyear": anchor_year, "endyear": anchor_year,
                                              "registrationkey": self.api_key}).json()
        base_cpi = float(b_res['Results']['series'][0]['data'][0]['value'])

        # get current year (2026) CPI
        n_res = requests.post(self.url,
                              json={"seriesid": [cpi_id], "startyear": str(curr_year - 1), "endyear": str(curr_year),
                                    "registrationkey": self.api_key}).json()
        now_cpi = float(n_res['Results']['series'][0]['data'][0]['value'])

        # Predicted formula: (Hourly wage * 2080) * (Current CPI / Benchmark CPI) * Technology premium (1.03)
        predicted = (base_h * 2080) * (now_cpi / base_cpi) * 1.03

        return {
            "role_name": role_name,
            "series_id": series_id,
            "anchor_year": anchor_year,
            "hourly_base": base_h,
            "predicted_salary_2026": round(predicted, 2)
        }