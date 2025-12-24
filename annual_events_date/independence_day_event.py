import pandas as pd

def independence_day(date):
    year = date.year

    independence_day_start = pd.Timestamp(year, 8, 16) #start
    independence_day_end = pd.Timestamp(year, 7, 18) #end

    return independence_day_start <= date <= independence_day_end