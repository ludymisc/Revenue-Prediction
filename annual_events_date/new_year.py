import pandas as pd

def new_year(date):
    year = date.year

    new_year_eve = pd.Timestamp(year, 12, 31)
    new_year_end = pd.Timestamp(year, 1, 2)

    return new_year_eve <= date <= new_year_end