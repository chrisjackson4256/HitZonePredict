import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def get_batter_zone_data(df):

    df = df[['batter', 'hit_zone']]
    df.set_index('batter', inplace=True)

    # one-hot-encode the hit_zone column
    ohe = OneHotEncoder()
    zones_ohe = pd.DataFrame(ohe.fit_transform(df).toarray())
    zones_ohe.index = df.index
    zones_ohe.columns = ['zone_1', 'zone_2', 'zone_3', 'zone_4']
    zones_ohe.reset_index(inplace=True, drop=False)

    # group by batter ID and sum
    df = zones_ohe.groupby('batter').sum()

    # finally, turn the pitch counts into percentages
    df_pct = pd.DataFrame()
    for i in range(len(df)):
        temp_df = pd.DataFrame(df.iloc[i]).T
        pitch_sum = sum(temp_df.iloc[0,:])
        temp_df.iloc[0,:] = temp_df.iloc[0,:] / pitch_sum
        df_pct = df_pct.append(temp_df)

    df_pct.reset_index(inplace=True, drop=False)
    df_pct.columns = ['batter', 'batter_zone_1', 'batter_zone_2', 'batter_zone_3', 'batter_zone_4']

    return df_pct
