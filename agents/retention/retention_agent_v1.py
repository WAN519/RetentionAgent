"""
agents/retention/retention_agent_v1.py

Agent 2 — Cox Proportional Hazard Attrition Risk Scorer.

This agent consumes the outputs of the Equity Agent and produces an
employee-level attrition risk score using a pre-trained Cox Survival model.

Pipeline:
  1. Load HR snapshot from CSV (features + Attrition ground truth label).
  2. Pull salary data from MongoDB collection: Salary.
  3. Pull equity gap data from MongoDB collection: Equity_Predictions.
  4. Merge all three data sources on employee_id.
  5. Engineer features (overtime flag, months since promotion, income scale).
  6. Run Cox model inference → partial hazard score per employee.
  7. Rank scores into percentiles and bucket into Low / Mid / High risk tiers.
  8. Apply deterministic rule override: employees who simultaneously have
       pay_gap ≤ -10%, MonthsSincePromotion ≥ 18, and active overtime
       are escalated to High regardless of the model score.
  9. Annotate each employee with human-readable risk reasons.
  10. Upsert results to MongoDB collection: Retention_Predictions.

MongoDB output schema (Retention_Predictions):
    {
        employee_id, analysis_date, base_pay, pay_gap,
        months_since_promotion, risk_score, risk_pct,
        risk_bucket, rule_flag, risk_reasons,
        model_version, status
    }
"""

import os
import json
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd

from tools.mongoDB import MarketDB

warnings.filterwarnings("ignore")


def ensure_columns(df: pd.DataFrame, needed_cols: list, fill_value=0) -> pd.DataFrame:
    """
    Add any missing columns to a DataFrame, filled with a default value.

    Used to align inference-time data to the training-time feature schema
    when certain optional columns are absent from the input CSV.

    Args:
        df (pd.DataFrame): Input DataFrame to patch.
        needed_cols (list): List of column names that must be present.
        fill_value: Value to use for missing columns (default: 0).

    Returns:
        pd.DataFrame: The same DataFrame with missing columns added in place.
    """
    for col in needed_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


class RetentionAgentV1:
    """
    Attrition risk scorer based on a Cox Proportional Hazard model.

    The Cox model was trained on IBM HR data with survival analysis framing:
      - `duration` = YearsAtCompany (time to event)
      - `event`    = Attrition == 'Yes' (did the employee leave?)

    The partial hazard output is a relative risk score; higher values indicate
    a higher probability of leaving compared to the baseline employee.

    Attributes:
        cph: Loaded CoxPH model (lifelines, deserialized from pickle).
        feature_columns (list[str]): Ordered feature list loaded from the JSON metadata file.
        db (MarketDB): Connected MongoDB handler for reading inputs and writing results.
    """

    def __init__(self, model_path: str, feature_json_path: str):
        """
        Load the Cox model and feature metadata, then connect to MongoDB.

        Args:
            model_path (str): Path to the serialized CoxPH model (.pkl).
            feature_json_path (str): Path to JSON file containing {"feature_columns": [...]}.

        Raises:
            FileNotFoundError: If either file does not exist.
            RuntimeError: If the MongoDB connection fails.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_json_path):
            raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")

        with open(model_path, "rb") as f:
            self.cph = pickle.load(f)

        with open(feature_json_path, "r") as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]

        self.db = MarketDB()
        if not self.db.is_connected:
            raise RuntimeError("MongoDB not connected. Check config.env.")

    def _load_salary(self) -> pd.DataFrame:
        """
        Load employee salary data from the MongoDB `Salary` collection.

        Normalizes column names to `employee_id` and `base_pay` regardless
        of the actual field names stored in the collection.

        Supported ID columns: employee_id, EmployeeNumber
        Supported salary columns: base_pay, current_salary, salary, MonthlyIncome

        Returns:
            pd.DataFrame: Two-column DataFrame [employee_id, base_pay],
                          or an empty DataFrame if the collection is empty.
        """
        docs = list(self.db.db["Salary"].find({}, {"_id": 0}))
        if not docs:
            print("Warning: Salary collection is empty.")
            return pd.DataFrame()

        salary_df = pd.DataFrame(docs)

        # Normalize the employee ID column name
        if "employee_id" not in salary_df.columns and "EmployeeNumber" in salary_df.columns:
            salary_df["employee_id"] = salary_df["EmployeeNumber"]

        # Normalize the salary column to a single `base_pay` field
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
        Load pay gap data from the MongoDB `Equity_Predictions` collection.

        The pay_gap field is expressed as a decimal ratio (e.g. -0.12 = -12%),
        suitable for direct use in the rule-based override logic.

        Returns:
            pd.DataFrame: Two-column DataFrame [employee_id, pay_gap],
                          or an empty DataFrame if the collection is empty.
        """
        docs = list(self.db.db["Equity_Predictions"].find({}, {"_id": 0}))
        if not docs:
            print("Warning: Equity_Predictions collection is empty.")
            return pd.DataFrame()

        equity_df = pd.DataFrame(docs)

        if "employee_id" not in equity_df.columns and "EmployeeNumber" in equity_df.columns:
            equity_df["employee_id"] = equity_df["EmployeeNumber"]

        # Handle nested dict format: equity_gaps.external_gap_pct
        if "equity_gaps" in equity_df.columns:
            def get_ext_gap_pct(x):
                try:
                    return float(x.get("external_gap_pct"))
                except Exception:
                    return np.nan

            equity_df["external_gap_pct"] = equity_df["equity_gaps"].apply(get_ext_gap_pct)

        # Convert percentage (e.g. -12.0) to decimal ratio (e.g. -0.12)
        if "external_gap_pct" in equity_df.columns:
            equity_df["pay_gap"] = pd.to_numeric(equity_df["external_gap_pct"], errors="coerce") / 100.0
        elif "pay_gap" not in equity_df.columns:
            equity_df["pay_gap"] = np.nan

        keep_cols = [c for c in ["employee_id", "pay_gap"] if c in equity_df.columns]
        return equity_df[keep_cols].copy() if keep_cols else pd.DataFrame()

    def _prepare_hr_data(self, hr_csv_path: str) -> pd.DataFrame:
        """
        Load and engineer features from the HR snapshot CSV.

        Engineered columns added:
          - OverTime_flag (int 0/1): binary from the 'OverTime' yes/no column
          - MonthsSincePromotion (float): YearsSinceLastPromotion * 12
          - MonthlyIncome_k (float): MonthlyIncome / 1000 for scale normalization
          - duration (float): YearsAtCompany, with 0 replaced by 0.5 to avoid
                              survival model degenerate behavior at t=0
          - event (int 0/1): 1 if Attrition == 'Yes', else 0

        Args:
            hr_csv_path (str): Path to the HR CSV file.

        Returns:
            pd.DataFrame: Enriched HR DataFrame ready for model input construction.

        Raises:
            FileNotFoundError: If the CSV path does not exist.
            ValueError: If neither 'employee_id' nor 'EmployeeNumber' column is found.
        """
        if not os.path.exists(hr_csv_path):
            raise FileNotFoundError(f"HR CSV not found: {hr_csv_path}")

        hr = pd.read_csv(hr_csv_path)

        if "employee_id" not in hr.columns:
            if "EmployeeNumber" in hr.columns:
                hr["employee_id"] = hr["EmployeeNumber"]
            else:
                raise ValueError("HR CSV must contain 'employee_id' or 'EmployeeNumber'.")

        # Binary overtime flag (the Cox model uses this as a numeric feature)
        if "OverTime" in hr.columns:
            hr["OverTime_flag"] = (hr["OverTime"].astype(str).str.lower() == "yes").astype(int)
        else:
            hr["OverTime_flag"] = 0

        # Convert years to months for finer-grained promotion recency signal
        if "YearsSinceLastPromotion" in hr.columns:
            hr["MonthsSincePromotion"] = hr["YearsSinceLastPromotion"].fillna(0).astype(float) * 12
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

        # Survival model requires duration > 0; replace 0-year tenure with 0.5
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
        Construct the final model input matrix aligned to the training feature schema.

        Optionally one-hot encodes Department and JobRole if those dummy columns
        are present in the saved feature list (indicates they were encoded at training time).

        Args:
            df (pd.DataFrame): Merged HR + salary + equity DataFrame.

        Returns:
            pd.DataFrame: Numeric DataFrame with exactly `self.feature_columns` columns,
                          NaNs filled with 0.
        """
        X = df.copy()

        # Detect if the model expects one-hot encoded categorical columns
        need_department = any(col.startswith("Department_") for col in self.feature_columns)
        need_jobrole = any(col.startswith("JobRole_") for col in self.feature_columns)

        dummy_cols = []
        if need_department and "Department" in X.columns:
            dummy_cols.append("Department")
        if need_jobrole and "JobRole" in X.columns:
            dummy_cols.append("JobRole")

        if dummy_cols:
            X = pd.get_dummies(X, columns=dummy_cols, drop_first=True)

        # Pad any columns present in training but absent in this dataset
        X = ensure_columns(X, self.feature_columns, fill_value=0)
        X_final = X[self.feature_columns].copy()

        for col in X_final.columns:
            X_final[col] = pd.to_numeric(X_final[col], errors="coerce")

        return X_final.fillna(0)

    def run(self, hr_csv_path: str):
        """
        Execute the full retention scoring pipeline and write results to MongoDB.

        Risk bucketing logic:
          - Percentile 0–50%  → Low
          - Percentile 50–90% → Mid
          - Percentile 90–100% → High
          - Rule override: any employee meeting all three conditions below is
            escalated to High regardless of model percentile:
              * pay_gap ≤ -10% (significantly underpaid vs. market)
              * MonthsSincePromotion ≥ 18 (no promotion in 1.5+ years)
              * OverTime_flag == 1 (currently working overtime)

        Args:
            hr_csv_path (str): Path to the HR snapshot CSV.
        """
        hr = self._prepare_hr_data(hr_csv_path)

        salary_df = self._load_salary()
        equity_df = self._load_equity_predictions()

        df = hr.copy()

        # Normalize employee_id to string to prevent int/str merge key mismatch
        df["employee_id"] = df["employee_id"].astype(str)

        if not salary_df.empty and "employee_id" in salary_df.columns:
            salary_df["employee_id"] = salary_df["employee_id"].astype(str)

        if not equity_df.empty and "employee_id" in equity_df.columns:
            equity_df["employee_id"] = equity_df["employee_id"].astype(str)

        # Left-join: employees without salary/equity data keep NaN for those fields
        if not salary_df.empty and "employee_id" in salary_df.columns:
            df = df.merge(salary_df, on="employee_id", how="left")

        if not equity_df.empty and "employee_id" in equity_df.columns:
            df = df.merge(equity_df, on="employee_id", how="left")

        if "base_pay" not in df.columns:
            df["base_pay"] = np.nan
        if "pay_gap" not in df.columns:
            df["pay_gap"] = np.nan

        X_final = self._build_model_input(df)

        # Cox partial hazard: higher value = higher relative risk of attrition
        df["risk_score"] = self.cph.predict_partial_hazard(X_final).values.ravel()

        # Convert to percentile rank within this cohort for interpretable bucketing
        df["risk_pct"] = df["risk_score"].rank(pct=True)

        df["risk_bucket_model"] = pd.cut(
            df["risk_pct"],
            bins=[0, 0.5, 0.9, 1.0],
            labels=["Low", "Mid", "High"],
            include_lowest=True
        )

        # Rule-based override: flag employees showing all three high-risk signals
        df["rule_flag"] = (
            (df["pay_gap"].fillna(0) <= -0.10) &
            (df["MonthsSincePromotion"] >= 18) &
            (df["OverTime_flag"] == 1)
        ).astype(int)

        # Merge model bucket with rule override — rule can only escalate, never downgrade
        df["risk_bucket_final"] = np.where(
            (df["risk_bucket_model"].astype(str) == "High") | (df["rule_flag"] == 1),
            "High",
            df["risk_bucket_model"].astype(str)
        )

        # Build human-readable reason list per employee for the HR dashboard
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

        # Upsert each employee's result into MongoDB
        out_coll = self.db.db["Retention_Predictions"]
        now = datetime.datetime.utcnow()

        for _, row in df.iterrows():
            emp_id = row.get("employee_id")

            # Preserve original ID type (int preferred; fall back to string)
            try:
                employee_key = int(emp_id) if pd.notna(emp_id) else None
            except Exception:
                employee_key = str(emp_id)

            doc = {
                "employee_id":           employee_key,
                "analysis_date":         now,
                "base_pay":              float(row["base_pay"]) if pd.notna(row["base_pay"]) else None,
                "pay_gap":               float(row["pay_gap"]) if pd.notna(row["pay_gap"]) else None,
                "months_since_promotion": float(row["MonthsSincePromotion"]),
                "risk_score":            float(row["risk_score"]),
                "risk_pct":              float(row["risk_pct"]),
                "risk_bucket":           str(row["risk_bucket_final"]),
                "rule_flag":             int(row["rule_flag"]),
                "risk_reasons":          row["risk_reasons"],
                "model_version":         "cox_retention_v1",
                "status":                "READY_FOR_ORCHESTRATOR"
            }

            out_coll.update_one(
                {"employee_id": doc["employee_id"]},
                {"$set": doc},
                upsert=True
            )

        self.db.close()
        print("Retention Agent complete. Results written to MongoDB: Retention_Predictions")


if __name__ == "__main__":
    # Resolve paths relative to the project root, regardless of where the script is called from
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    MODEL_PATH        = os.path.join(BASE_DIR, "models", "cox_retention_v1.pkl")
    FEATURE_JSON_PATH = os.path.join(BASE_DIR, "models", "cox_retention_v1_features.json")
    HR_CSV_PATH       = os.path.join(BASE_DIR, "data", "Attrition.csv")

    print("Model:    ", MODEL_PATH)
    print("Features: ", FEATURE_JSON_PATH)
    print("HR CSV:   ", HR_CSV_PATH)

    agent = RetentionAgentV1(MODEL_PATH, FEATURE_JSON_PATH)
    agent.run(HR_CSV_PATH)