import pandas as pd 

def easter_week(date):
    year = date.year

    easter_start = pd.Timestamp(year, 4, 1) #start
    easter_end = pd.Timestamp(year, 4, 10) #end

    return easter_start <= date <= easter_end
