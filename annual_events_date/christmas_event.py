import pandas as pd

def christmas_event(date):
    year = date.year

    christmas_start = pd.Timestamp(year, 12, 17)
    christmas_end = pd.Timestamp(year, 12, 31)

    return christmas_start <= date <= christmas_end