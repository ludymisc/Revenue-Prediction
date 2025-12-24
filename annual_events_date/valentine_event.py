import pandas as pd

def valentine_event(date):
    year = date.year

    valentine_event_start = pd.Timestamp(year, 2, 7) #start
    valentine_event_end = pd.Timestamp(year, 2, 21) #end

    return valentine_event_start <= date <= valentine_event_end
