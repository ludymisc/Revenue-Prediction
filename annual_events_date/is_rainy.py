import pandas as pd

def is_rainy_season(dt):
    year = dt.year

    rainy_start_1 = pd.Timestamp(year, 10, 1)
    rainy_end_1   = pd.Timestamp(year, 12, 31)

    rainy_start_2 = pd.Timestamp(year, 1, 1)
    rainy_end_2   = pd.Timestamp(year, 3, 31)

    return (
        rainy_start_1 <= dt <= rainy_end_1
        or
        rainy_start_2 <= dt <= rainy_end_2
    )