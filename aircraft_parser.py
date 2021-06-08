import json as js
import pandas as pd
import requests as rq

def execute_request(API_key, entrypoint, offset):
    url = entrypoint + f"&offset={offset}"
    r = rq.get(url)
    j = r.json()
    data = j['data']
    df = pd.DataFrame(data)
    return df

API_key = "YOUR_API_KEY"  # если хочешь, можешь получить свой ключ на сайте
entrypoint = f"http://api.aviationstack.com/v1/airplanes?access_key={API_key}"

for i in range(190):
    try:
        df = execute_request(API_key=API_key, entrypoint=entrypoint, offset=i*100)
    except:
        print(i)
    if i == 0:
        df_fetched = df.copy()
    else:
        df_fetched = df_fetched.append(df, ignore_index=True)

df_fetched.to_csv("aircraft_data.csv", sep=';')