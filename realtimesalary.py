from apify_client import ApifyClient
import numpy as np

# 初始化（去 Apify Console -> Settings -> Integrations 拿你的 Token）
client = ApifyClient("YOUR_APIFY_TOKEN")

def get_realtime_seattle_median(job_title="Software Engineer"):
    # 准备输入：针对 Levels.fyi 或类似的 Scraper
    # 这里的 actor id 可以在 Apify Store 找到，例如：consummate_mandala/levels-fyi-salary-scraper
    run_input = {
        "location": "Seattle, WA",
        "jobTitle": job_title,
        "maxResults": 50  # 拿 50 条最近的实时数据就足够算中位数了
    }

    # 运行 Actor
    print(f"🚀 正在通过 Apify 抓取西雅图 {job_title} 的实时薪资...")
    run = client.actor("consummate_mandala/levels-fyi-salary-scraper").call(run_input=run_input)

    # 从结果集中提炼数据
    salaries = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        # Levels.fyi 通常返回 totalCompensation 或 baseSalary
        salary_str = item.get("totalCompensation", "0")
        # 简单清洗：去掉 $ 和 ,
        price = float(salary_str.replace('$', '').replace(',', ''))
        if price > 0:
            salaries.append(price)

    if salaries:
        median = np.median(salaries)
        print(f"✅ 成功！实时中位数: ${median:,.2f}")
        return median
    return None

# 测试运行
# get_realtime_seattle_median("Data Scientist")