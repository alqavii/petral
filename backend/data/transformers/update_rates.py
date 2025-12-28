import os
from rates_adapter import RatesAdapter

if __name__ == "__main__":
    if not os.getenv("FRED_API_KEY"):
        print("Error: FRED_API_KEY not set")
        exit(1)

    RatesAdapter.updateRates()
    last = RatesAdapter.getLastUpdateDate()
    print(f"Last update: {last}")
