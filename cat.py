import joblib

# 1. 加载你的模型
model = joblib.load('notebook/Model/agent_salary_regressor.pkl')

# 2. 提取模型要求的特征顺序
if hasattr(model, 'feature_name_'):
    print("--- ⚖️ 模型要求的特征顺序 (复制下面的列表) ---")
    print(model.feature_name_)
    
# 3. 提取模型要求的类别特征
if hasattr(model, 'categorical_feature'):
    print("\n--- 🏷️ 类别特征索引 ---")
    print(model.categorical_feature)
    print("test")