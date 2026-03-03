import os
import joblib
import pandas as pd
import datetime
import urllib.parse
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from market_db import MarketDB  # 确保你之前的 market_db.py 在同级目录

class EquityAgent:
    def __init__(self, model_path, config_path="config.env"):
        """
        初始化第一个 Agent: 薪酬公平性智能体
        :param model_path: 训练好的 LightGBM 模型路径 (.pkl 或 .txt)
        """
        # 1. 加载模型
        try:
            self.model = joblib.load(model_path)
            print(f"LightGBM Model Loaded: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        # 2. 连接数据库
        self.db = MarketDB()
        
        # 3. 定义训练时严格的特征顺序（必须与你训练模型时完全一致）
        self.feature_columns = [
            'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 
            'EducationField', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 
            'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 
            'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears', 
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
            'YearsWithCurrManager', 'Market_Median_2026', 'Internal_Salary_Rank', 
            'Performance_Consistency'
        ]

    def run_analysis_pipeline(self, csv_input_path):
        """
        执行完整的分析流水线
        """
        if not self.db.is_connected:
            print("Database not connected. Aborting.")
            return

        # 读取员工测试数据
        df = pd.read_csv(csv_input_path)
        print(f"Processing {len(df)} employees...")

        for index, row in df.iterrows():
            emp_id = row.get('employee_id', f"UNKNOWN_{index}")
            role = row.get('role_name')
            actual_salary = row.get('current_salary')

            # --- STEP A: 获取外部市场实时数据 (Perception) ---
            market_data = self.db.get_benchmark(role)
            if not market_data:
                print(f"Skipping {emp_id}: No market data found for {role} in MongoDB.")
                continue
            
            market_val = market_data['predicted_salary_2026']

            # 1. 初始化特征字典并注入动态数据
            input_dict = row.to_dict()
            input_dict['Market_Median_2026'] = market_val
            input_dict['Internal_Salary_Rank'] = row.get('Internal_Salary_Rank', 0.5)
            input_dict['Performance_Consistency'] = row.get('Performance_Consistency', row['PerformanceRating'])

            # 2. 构造 DataFrame 并强制排序 (使用你刚发给我的列表)
            trained_features = [
                'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 
                'EducationField', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 
                'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 
                'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears', 
                'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
                'YearsWithCurrManager', 'Market_Median_2026', 'Internal_Salary_Rank', 
                'Performance_Consistency'
            ]
            
            final_input = pd.DataFrame([input_dict])[trained_features]

            # 3. 强制类型对齐 (根据你之前的 info() 结果)
            # 类别型列
            cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Over18']
            for col in cat_cols:
                final_input[col] = final_input[col].astype('category')
            
            # 浮点型列
            float_cols = ['Market_Median_2026', 'Internal_Salary_Rank', 'Performance_Consistency']
            for col in float_cols:
                final_input[col] = final_input[col].astype('float64')

            # 整数型列 (剩下的所有)
            int_cols = [c for c in trained_features if c not in cat_cols + float_cols]
            for col in int_cols:
                final_input[col] = pd.to_numeric(final_input[col], errors='coerce').fillna(0).astype('int64')

            # --- STEP C: 执行内部公平性预测 (LightGBM Inference) ---
            internal_valuation = self.model.predict(final_input)[0]

            # --- STEP D: 计算公平性缺口 (Gap Analysis) ---
            external_gap = (actual_salary - market_val) / market_val
            internal_gap = (actual_salary - internal_valuation) / internal_valuation

            # --- STEP E: 结果持久化至 MongoDB (Blackboard Storage) ---
            analysis_result = {
                "employee_id": emp_id,
                "role_name": role,
                "analysis_date": datetime.datetime.utcnow(),
                "actual_salary": round(actual_salary, 2),
                "benchmarks": {
                    "market_2026_target": round(market_val, 2),
                    "internal_fair_valuation": round(internal_valuation, 2)
                },
                "equity_gaps": {
                    "external_gap_pct": round(external_gap * 100, 2),
                    "internal_gap_pct": round(internal_gap * 100, 2)
                },
                "status": "READY_FOR_ORCHESTRATOR"
            }

            # 写入 Equity_Predictions 集合，供最后的 LLM Agent 提取
            try:
                self.db.db["Equity_Predictions"].update_one(
                    {"employee_id": emp_id},
                    {"$set": analysis_result},
                    upsert=True
                )
                print(f"Success: {emp_id} | IntGap: {analysis_result['equity_gaps']['internal_gap_pct']}% | ExtGap: {analysis_result['equity_gaps']['external_gap_pct']}%")
            except PyMongoError as e:
                print(f"DB Write Error for {emp_id}: {e}")

        print("\n--- Agent 1: Analysis Cycle Complete ---")
        self.db.close()

if __name__ == "__main__":
    # 填入你的模型文件名
    MODEL_FILE = "notebook/Model/agent_salary_regressor.pkl" 
    # 填入你的测试数据文件名
    TEST_DATA = "ibm_enhanced_test.csv"

    if os.path.exists(MODEL_FILE) and os.path.exists(TEST_DATA):
        agent = EquityAgent(model_path=MODEL_FILE)
        agent.run_analysis_pipeline(TEST_DATA)
    else:
        print("Please check if your model file and test CSV exist.")