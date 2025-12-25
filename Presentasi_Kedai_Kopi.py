#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from annual_events_date.christmas_event import christmas_event
from annual_events_date.easter_event import easter_week
from annual_events_date.new_year import new_year
from annual_events_date.valentine_event import valentine_event
from annual_events_date.is_rainy import is_rainy_season
from annual_events_date.halloween_event import halloween
from annual_events_date.independence_day_event import independence_day
from annual_events_date.is_black_friday import is_black_friday_promo
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



# In[4]:


np.random.seed(42) #karena semua individu data engineering menyayangi angka 42

event_func = [
    christmas_event,
    easter_week,
    new_year,
    valentine_event,
    halloween,
    independence_day,
    is_black_friday_promo
]

rows = 500

df = pd.DataFrame(
    {
        "datetime" : pd.date_range(start = "2023-01-12", periods=rows, freq = "D"),
        "event" : False,
        "promo" : 0,
        "kedai_ramai" : False,
        "is_rain" : False,
        # "pegawai_lengkap" : True,
        "revenue" : 1_000_000
    }
)

#apply func to event
event_mask = np.zeros(rows, dtype=bool)
for events in event_func:
    event_mask |= df["datetime"].apply(events)
df['event'] = event_mask

#apply func to promo encode
df.loc[df['event'], 'promo'] = np.random.randint(2,4, size=df['event'].sum())
promo_mask_event = df['promo'] >= 2 
promo_mask_normal = df['promo'] < 2 

#apply func to kedai_ramai
df.loc[promo_mask_event, 'kedai_ramai'] = np.random.rand(promo_mask_event.sum()) < 0.8
df.loc[promo_mask_normal, 'kedai_ramai'] = np.random.rand(promo_mask_normal.sum()) < 0.55 

#apply func to is_rain
rainy_season_mask = df["datetime"].apply(is_rainy_season)
df['is_rain'] = rainy_season_mask.apply(
    lambda x : np.random.rand() < (0.9 if x else 0.1)
)

#pegawai_lengkap rand
#will add in the future

#revenue affect func
base_revenue = 10.0

df["rev_score"] = 1.0
# promo
df.loc[df["promo"] == 2, "rev_score"] += 0.15
df.loc[df["promo"] == 3, "rev_score"] += 0.25

# kedai ramai
df.loc[df["kedai_ramai"], "rev_score"] += 0.30

# hujan
df.loc[df["is_rain"], "rev_score"] -= 0.10
# event + ramai = peak day
df.loc[df["event"] & df["kedai_ramai"], "rev_score"] += 0.20

# hujan + sepi = sekarat
df.loc[df["is_rain"] & ~df["kedai_ramai"], "rev_score"] -= 0.20
df["rev_score"] = df["rev_score"].clip(0.5, 2.5)
df["revenue"] = base_revenue * df["rev_score"]
df["revenue"] *= np.random.normal(1.0, 0.05, size=len(df))

# df.iloc[100:200,:]
df.iloc[80:100, :]


# In[5]:


X = df.drop(['revenue', "rev_score", "datetime", "event"], axis = 1)
y = df["revenue"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)
categorical_cols =  X_train.select_dtypes(include=["bool"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include = np.number).columns.tolist()

encoder = OrdinalEncoder(handle_unknown = "ignore")
model = RandomForestRegressor(n_estimators = 100, random_state = 0)

preprocessor = ColumnTransformer(
    transformers = [
        ("num", "passthrough", numerical_cols),
        ("cat", "passthrough", categorical_cols)    ]
)

pipeline = Pipeline(
    steps = [
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
print("MAE: ", mean_absolute_error(y_valid, preds))
print(X_valid)
print(y_valid)
print(mean_absolute_error(y_valid, preds)/df["revenue"].mean() * 100, "%")



# In[6]:


import joblib

joblib.dump(pipeline, "revenue_model_3_withour_event.pkl")


# In[7]:


tot_rev = df["revenue"].sum()
print("total revenue :", tot_rev)

