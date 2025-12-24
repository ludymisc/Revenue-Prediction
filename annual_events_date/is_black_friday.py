import pandas as pd

def is_black_friday_promo(dt):
    year = dt.year

    november = pd.date_range(start=f"{year}-11-01", end=f"{year}-11-30")
    fridays = november[november.weekday == 4]  # 0=Mon ... 4=Fri
    last_friday = fridays[-1]
    
    return last_friday <= dt <= last_friday + pd.Timedelta(days=1)
