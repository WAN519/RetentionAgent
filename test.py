import pandas as pd
import numpy as np

def generate_enhanced_csv(file_name="ibm_enhanced_test.csv"):
    # 1. 定义你的模型需要的 25 个特征（严格顺序）
    features = [
        'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 
        'EducationField', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 
        'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 
        'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears', 
        'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsWithCurrManager', 'Market_Median_2026', 'Internal_Salary_Rank', 
        'Performance_Consistency'
    ]

    # 2. 模拟 5 名员工的数据
    num_samples = 5
    
    data = {
        # --- 管理字段 (不进入模型，但 Agent 需要用) ---
        'employee_id': [f'EMP{i:03d}' for i in range(1, num_samples + 1)],
        'role_name': [
            'Software_Developer_Seattle', 
            'Data_Scientist_Seattle', 
            'Software_Developer_Seattle', 
            'Data_Scientist_Seattle', 
            'Software_Developer_Seattle'
        ],
        'current_salary': [145000, 168000, 132000, 185000, 155000],

        # --- 模型特征字段 (模拟 IBM 编码后的数值) ---
        'BusinessTravel': [1, 2, 1, 1, 2],
        'DailyRate': [1102, 1218, 1373, 890, 591],
        'Department': [1, 1, 1, 1, 1],
        'DistanceFromHome': [1, 2, 2, 7, 2],
        'Education': [2, 4, 2, 4, 1],
        'EducationField': [1, 1, 2, 1, 1],
        'HourlyRate': [94, 52, 92, 56, 40],
        'JobInvolvement': [3, 2, 2, 3, 3],
        'JobLevel': [2, 3, 1, 4, 2],
        'JobRole': [7, 2, 7, 2, 7],
        'MaritalStatus': [2, 1, 2, 1, 1],
        'MonthlyRate': [19479, 24907, 2396, 23159, 12290],
        'NumCompaniesWorked': [8, 1, 1, 1, 9],
        'Over18': [1, 1, 1, 1, 1], # IBM 数据集中通常全为 1
        'PercentSalaryHike': [11, 23, 15, 11, 12],
        'PerformanceRating': [3, 4, 3, 3, 3],
        'StockOptionLevel': [0, 1, 0, 1, 1],
        'TotalWorkingYears': [8, 10, 7, 12, 6],
        'TrainingTimesLastYear': [0, 3, 3, 3, 3],
        'YearsAtCompany': [6, 5, 0, 8, 2],
        'YearsInCurrentRole': [4, 2, 0, 7, 2],
        'YearsWithCurrManager': [0, 3, 0, 0, 2],
        
        # --- 增强特征 (Agent 运行前可先填占位符，Agent 1 会自动动态覆盖 Market_Median_2026) ---
        'Market_Median_2026': [0, 0, 0, 0, 0], 
        'Internal_Salary_Rank': [0.5, 0.8, 0.3, 0.9, 0.4],
        'Performance_Consistency': [3, 4, 3, 3, 3]
    }

    df = pd.DataFrame(data)
    
    # 确保保存时列的完整性
    df.to_csv(file_name, index=False)
    print(f"✅ Enhanced test CSV generated: {file_name}")
    print(f"📊 Features included: {len(features)} | Total columns: {len(df.columns)}")

if __name__ == "__main__":
    generate_enhanced_csv()