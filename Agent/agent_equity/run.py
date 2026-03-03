import os
from Agent.agent_equity.market_db import MarketDB
from Agent.agent_equity.market_logic import MarketDataCoordinator

def main():
    # 1. Initialize the database
    db = MarketDB()
    if not db.is_connected:
        return

    # 2. run
    api_key = os.getenv("BLS_API_KEY")
    coordinator = MarketDataCoordinator(db, api_key)

    # 3. set job in seattle
    targets = [
        ("OEUM004266000000015125203", "Software_Developer_Seattle"),
        ("OEUM004266000000015205103", "Data_Scientist_Seattle")
    ]

    print("\n--- RetentionAgent Market Sync Starting ---")
    for sid, name in targets:
        result = coordinator.get_market_intelligence(sid, name)
        if result:
            print(f"{name} | Predicted Market: ${result['predicted_salary_2026']:,}")

    # 4. close mangodb
    db.close()
    print("--- Sync Finished ---")

if __name__ == "__main__":
    main()