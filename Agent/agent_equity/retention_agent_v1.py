import os
import json
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd

from Agent.agent_equity.market_db import MarketDB

warnings.filterwarnings("ignore")


def ensure_columns(df: pd.DataFrame, needed_cols: list, fill_value=0):
    """Ensure all required columns exist in df."""
    for col in needed_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


class RetentionAgentV1:
    """
    Agent2 V1: Snapshot Cox Survival Risk Scoring

    Pipeline:
    1. Read HR snapshot CSV
    2. Read Salary from MongoDB
    3. Read Equity_Predictions from MongoDB
    4. Merge data
    5. Build model features
    6. Predict attrition risk using trained Cox model
    7. Write results to MongoDB: Retention_Predictions
    """

    def __init__(self, model_path: str, feature_json_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(feature_json_path):
            raise FileNotFoundError(f"Feature JSON file not found: {feature_json_path}")

        with open(model_path, "rb") as f:
            self.cph = pickle.load(f)

        with open(feature_json_path, "r") as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]

        self.db = MarketDB()
        if not self.db.is_connected:
            raise RuntimeError("MongoDB not connected. Please check config.env")

    def _load_salary(self) -> pd.DataFrame:
        """
        Load salary data from MongoDB collection: Salary

        Supported employee key columns:
        - employee_id
        - EmployeeNumber

        Supported salary columns:
        - base_pay
        - current_salary
        - salary
        - MonthlyIncome
        """
        docs = list(self.db.db["Salary"].find({}, {"_id": 0}))
        if not docs:
            print("Warning: Salary collection is empty.")
            return pd.DataFrame()

        salary_df = pd.DataFrame(docs)

        if "employee_id" not in salary_df.columns and "EmployeeNumber" in salary_df.columns:
            salary_df["employee_id"] = salary_df["EmployeeNumber"]

        if "base_pay" not in salary_df.columns:
            if "current_salary" in salary_df.columns:
                salary_df["base_pay"] = salary_df["current_salary"]
            elif "salary" in salary_df.columns:
                salary_df["base_pay"] = salary_df["salary"]
            elif "MonthlyIncome" in salary_df.columns:
                salary_df["base_pay"] = salary_df["MonthlyIncome"]

        keep_cols = [c for c in ["employee_id", "base_pay"] if c in salary_df.columns]
        return salary_df[keep_cols].copy() if keep_cols else pd.DataFrame()

    def _load_equity_predictions(self) -> pd.DataFrame:
        """
        Load equity predictions from MongoDB collection: Equity_Predictions

        Output:
        - employee_id
        - pay_gap   (ratio, e.g. -0.12 means -12%)
        """
        docs = list(self.db.db["Equity_Predictions"].find({}, {"_id": 0}))
        if not docs:
            print("Warning: Equity_Predictions collection is empty.")
            return pd.DataFrame()

        equity_df = pd.DataFrame(docs)

        if "employee_id" not in equity_df.columns and "EmployeeNumber" in equity_df.columns:
            equity_df["employee_id"] = equity_df["EmployeeNumber"]

        # Case 1: nested dict
        if "equity_gaps" in equity_df.columns:
            def get_ext_gap_pct(x):
                try:
                    return float(x.get("external_gap_pct"))
                except Exception:
                    return np.nan

            equity_df["external_gap_pct"] = equity_df["equity_gaps"].apply(get_ext_gap_pct)

        # Case 2: already flat
        if "external_gap_pct" in equity_df.columns:
            equity_df["pay_gap"] = pd.to_numeric(equity_df["external_gap_pct"], errors="coerce") / 100.0
        elif "pay_gap" not in equity_df.columns:
            equity_df["pay_gap"] = np.nan

        keep_cols = [c for c in ["employee_id", "pay_gap"] if c in equity_df.columns]
        return equity_df[keep_cols].copy() if keep_cols else pd.DataFrame()

    def _prepare_hr_data(self, hr_csv_path: str) -> pd.DataFrame:
        """Read HR CSV and create engineered columns."""
        if not os.path.exists(hr_csv_path):
            raise FileNotFoundError(f"HR CSV not found: {hr_csv_path}")

        hr = pd.read_csv(hr_csv_path)

        if "employee_id" not in hr.columns:
            if "EmployeeNumber" in hr.columns:
                hr["employee_id"] = hr["EmployeeNumber"]
            else:
                raise ValueError("HR CSV must contain either 'employee_id' or 'EmployeeNumber'")

        # Overtime flag
        if "OverTime" in hr.columns:
            hr["OverTime_flag"] = (hr["OverTime"].astype(str).str.lower() == "yes").astype(int)
        else:
            hr["OverTime_flag"] = 0

        # Keep fully aligned with training
        if "YearsSinceLastPromotion" in hr.columns:
            # If training used MonthsSincePromotion, keep this line
            hr["MonthsSincePromotion"] = hr["YearsSinceLastPromotion"].fillna(0).astype(float) * 12

            # Also keep original field available in case feature_columns expects it
            hr["YearsSinceLastPromotion"] = hr["YearsSinceLastPromotion"].fillna(0).astype(float)
        else:
            hr["MonthsSincePromotion"] = 0.0
            hr["YearsSinceLastPromotion"] = 0.0

        if "MonthlyIncome" in hr.columns:
            hr["MonthlyIncome"] = pd.to_numeric(hr["MonthlyIncome"], errors="coerce").fillna(0)
            hr["MonthlyIncome_k"] = hr["MonthlyIncome"] / 1000.0
        else:
            hr["MonthlyIncome"] = 0.0
            hr["MonthlyIncome_k"] = 0.0

        if "YearsAtCompany" in hr.columns:
            hr["duration"] = pd.to_numeric(hr["YearsAtCompany"], errors="coerce").fillna(1).replace(0, 0.5)
        else:
            hr["duration"] = 1.0

        if "Attrition" in hr.columns:
            hr["event"] = (hr["Attrition"].astype(str).str.lower() == "yes").astype(int)
        else:
            hr["event"] = 0

        return hr

    def _build_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build model input aligned exactly to training-time features.
        Supports optional Department / JobRole dummy features if present in feature_columns.
        """
        X = df.copy()

        need_department = any(col.startswith("Department_") for col in self.feature_columns)
        need_jobrole = any(col.startswith("JobRole_") for col in self.feature_columns)

        dummy_cols = []
        if need_department and "Department" in X.columns:
            dummy_cols.append("Department")
        if need_jobrole and "JobRole" in X.columns:
            dummy_cols.append("JobRole")

        if dummy_cols:
            X = pd.get_dummies(X, columns=dummy_cols, drop_first=True)

        X = ensure_columns(X, self.feature_columns, fill_value=0)
        X_final = X[self.feature_columns].copy()

        for col in X_final.columns:
            X_final[col] = pd.to_numeric(X_final[col], errors="coerce")

        X_final = X_final.fillna(0)
        return X_final

    def run(self, hr_csv_path: str):
        hr = self._prepare_hr_data(hr_csv_path)

        salary_df = self._load_salary()
        equity_df = self._load_equity_predictions()

        df = hr.copy()
        # -------------------------------
        # Normalize merge key type
        # Convert all employee_id columns to string to avoid int/object merge error
        # -------------------------------
        df["employee_id"] = df["employee_id"].astype(str)

        if not salary_df.empty and "employee_id" in salary_df.columns:
            salary_df["employee_id"] = salary_df["employee_id"].astype(str)

        if not equity_df.empty and "employee_id" in equity_df.columns:
            equity_df["employee_id"] = equity_df["employee_id"].astype(str)

        if not salary_df.empty and "employee_id" in salary_df.columns:
            df = df.merge(salary_df, on="employee_id", how="left")

        if not equity_df.empty and "employee_id" in equity_df.columns:
            df = df.merge(equity_df, on="employee_id", how="left")

        if "base_pay" not in df.columns:
            df["base_pay"] = np.nan

        if "pay_gap" not in df.columns:
            df["pay_gap"] = np.nan

        X_final = self._build_model_input(df)

        df["risk_score"] = self.cph.predict_partial_hazard(X_final).values.ravel()
        df["risk_pct"] = df["risk_score"].rank(pct=True)

        df["risk_bucket_model"] = pd.cut(
            df["risk_pct"],
            bins=[0, 0.5, 0.9, 1.0],
            labels=["Low", "Mid", "High"],
            include_lowest=True
        )

        df["rule_flag"] = (
            (df["pay_gap"].fillna(0) <= -0.10) &
            (df["MonthsSincePromotion"] >= 18) &
            (df["OverTime_flag"] == 1)
        ).astype(int)

        df["risk_bucket_final"] = np.where(
            (df["risk_bucket_model"].astype(str) == "High") | (df["rule_flag"] == 1),
            "High",
            df["risk_bucket_model"].astype(str)
        )

        reasons = []
        for _, row in df.iterrows():
            reason_list = []

            if pd.notna(row.get("pay_gap")) and row.get("pay_gap") <= -0.10:
                reason_list.append("pay gap below market")
            if row.get("MonthsSincePromotion", 0) >= 18:
                reason_list.append("long time since last promotion")
            if row.get("OverTime_flag", 0) == 1:
                reason_list.append("overtime")
            if row.get("JobSatisfaction", 99) <= 2:
                reason_list.append("low job satisfaction")
            if row.get("WorkLifeBalance", 99) <= 2:
                reason_list.append("low work-life balance")

            reasons.append(reason_list)

        df["risk_reasons"] = reasons

        out_coll = self.db.db["Retention_Predictions"]
        now = datetime.datetime.utcnow()

        for _, row in df.iterrows():
            emp_id = row.get("employee_id")

            try:
                employee_key = int(emp_id) if pd.notna(emp_id) else None
            except Exception:
                employee_key = str(emp_id)

            doc = {
                "employee_id": employee_key,
                "analysis_date": now,
                "base_pay": float(row["base_pay"]) if pd.notna(row["base_pay"]) else None,
                "pay_gap": float(row["pay_gap"]) if pd.notna(row["pay_gap"]) else None,
                "months_since_promotion": float(row["MonthsSincePromotion"]),
                "risk_score": float(row["risk_score"]),
                "risk_pct": float(row["risk_pct"]),
                "risk_bucket": str(row["risk_bucket_final"]),
                "rule_flag": int(row["rule_flag"]),
                "risk_reasons": row["risk_reasons"],
                "model_version": "cox_retention_v1",
                "status": "READY_FOR_ORCHESTRATOR"
            }

            out_coll.update_one(
                {"employee_id": doc["employee_id"]},
                {"$set": doc},
                upsert=True
            )

        self.db.close()
        print("✅ Agent2 V1 complete. Results written to MongoDB: Retention_Predictions")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    MODEL_PATH = os.path.join(BASE_DIR, "notebook", "Model", "cox_retention_v1.pkl")
    FEATURE_JSON_PATH = os.path.join(BASE_DIR, "notebook", "Model", "cox_retention_v1_features.json")
    HR_CSV_PATH = os.path.join(BASE_DIR, "Attrition.csv")

    print("Using model:", MODEL_PATH)
    print("Using features:", FEATURE_JSON_PATH)
    print("Using HR CSV:", HR_CSV_PATH)

    agent = RetentionAgentV1(MODEL_PATH, FEATURE_JSON_PATH)
    agent.run(HR_CSV_PATH)