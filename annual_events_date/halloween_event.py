import pandas as pd

def halloween(date):
    year = date.year

    halloween_start = pd.Timestamp(year, 10, 31) #start
    halloween_end = pd.Timestamp(year, 11, 7) #end


    return halloween_start <= date <= halloween_end