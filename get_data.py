from pybaseball import statcast, pitching_stats
import numpy as np
import pandas as pd
from pitch_clustering import pitch_clustering

def get_data(get_data_func, train_dates_list):
    '''
    General function to get data from statcast
    input: get_data_func - function (from below) to use to get data
           train_dates_list - list of start/end dates for data
    output: dataframe
    '''
    # build a list of dataframes (one for each year)
    data_list = []
    for dates in train_dates_list:
        df = get_data_func(start=dates[0], end=dates[1])
        data_list.append(df)

    # concat the list of dataframes into one large dataframe
    data = pd.concat(data_list)

    return data


def get_hit_zone(start, end):
    '''
    Function which calculates the "hit zone" for a specific at batter
    input: start - the start date of the statcast data
           end - the end date of the statcast data
    output: a dataframe with index columns (game ID, pitch ID, batter ID, pitcher ID)
            and the "hit zone"
    '''
    # get the data from statcast
    data = statcast(start_dt=start, end_dt=end, verbose=0)

    # just keep the "events" (i.e., the end result of an at-bat)
    data = data[~pd.isnull(data['events'])]

    # remove events that don't involve the ball being put into play in a meaningful way
    # (i.e., no strikeouts, walks, sacrifice bunts, caught stealing, hit by pitch, etc.)
    inplay_event_list = ['field_out', 'sac_fly', 'force_out', 'field_error', 'double', 'home_run',
                         'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'triple',
                         'double_play']
    data = data[data['events'].isin(inplay_event_list)]

    # select the columns to keep
    cols_to_keep = ['game_pk', 'index', 'batter', 'pitcher', 'events', 'hc_x', 'hc_y', 'hit_distance_sc']
    data = data[cols_to_keep]

    # make sure index columns are int's
    for col in ['game_pk', 'index', 'batter', 'pitcher']:
        data[col] = data[col].astype(int)

    # transform the hit location features such that home plate is at the origin
    data['hit_x'] = data['hc_x'].subtract(125.42)
    data['hit_y'] = data['hc_y'].multiply(-1).add(198.27)
    data.drop(['hc_x', 'hc_y'], axis=1, inplace=True)

    # use the coordinates of the hit to calculate the spray angle (in degrees)
    def spray_angle(x, y):
        return np.arctan(x/(y+0.001)) * (180 / 3.14)
    data['spray_angle'] = data.apply(lambda x: spray_angle(x.hit_x, x.hit_y), axis=1)

    # identify the zone of the hit:
    #   1 -> right side of the infield (spray angle > 0 and hit_distance < 50)
    #   2 -> left side of the infield (spray angle < 0 and hit_distance < 50)
    #   3 -> right side of the outfield (spray angle > 0 and hit_distance >= 50)
    #   4 -> left side of the outfield (spray angle < 0 and hit_distance >= 50)
    def hit_zone(angle, dist):
        if angle > 0 and dist < 50:
            return 1
        elif angle < 0 and dist < 50:
            return int(2)
        elif angle > 0 and dist >= 50:
            return 3
        elif angle < 0 and dist >= 50:
            return 4
    data['hit_zone'] = data.apply(lambda x: hit_zone(x.spray_angle, x.hit_distance_sc), axis=1)
    data = data[pd.notna(data['hit_zone'])]
    data['hit_zone'] = data['hit_zone'].astype(int)

    # return only the columns we need (index columns and the outcome "hit_zone")
    return data[['game_pk', 'index', 'batter', 'pitcher', 'hit_zone']]


def get_pitch_data(start, end):
    '''
    Function to extract pitch information from statcast data
    input: start - the start date of the statcast data
           end - the end date of the statcast data
    output: a dataframe with pitcher's name and ID as well as information on every
            pitch they've thrown
    '''
    data = statcast(start_dt=start, end_dt=end, verbose=0)

    pitch_cols = ['pitcher', 'player_name', 'release_speed', 'release_spin_rate', 'release_extension',
                  'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                  'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    data = data[pitch_cols]

    # drop rows with missing values
    data.dropna(inplace=True)

    # perform PCA and K-Means clustering
    print("Performing PCA and K-Means Clustering...")
    data = pitch_clustering(data)

    return data


def get_situation_data(start, end):

    data = statcast(start_dt=start, end_dt=end, verbose=0)

    # just keep the "events" (i.e., the end result of an at-bat)
    data = data[~pd.isnull(data['events'])]

    # remove events that don't involve the ball being put into play in a meaningful way
    # (i.e., no strikeouts, walks, sacrifice bunts, caught stealing, hit by pitch, etc.)
    inplay_event_list = ['field_out', 'sac_fly', 'force_out', 'field_error', 'double', 'home_run',
                         'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'triple',
                         'double_play']
    data = data[data['events'].isin(inplay_event_list)]

    # trim down the features
    cols_to_keep = ['game_pk', 'index', 'batter', 'pitcher', 'stand', 'p_throws', 'balls', 'strikes', 'outs_when_up',
                    'inning', 'on_1b', 'on_2b', 'on_3b', 'bat_score', 'fld_score']
    data = data[cols_to_keep]

    # make sure index columns are int's
    for col in ['game_pk', 'index', 'batter', 'pitcher']:
        data[col] = data[col].astype(int)

    data['bat_right'] = data['stand'].apply(lambda x: x == 'R')
    data['pitch_right'] = data['p_throws'].apply(lambda x: x == 'R')
    data.drop(['stand', 'p_throws'], axis=1, inplace=True)

    for col in ['on_1b', 'on_2b', 'on_3b']:
        data[col] = data[col].apply(lambda x: x == x)

    data['score_diff'] = data['bat_score'] - data['fld_score']
    data.drop(['bat_score', 'fld_score'], axis=1, inplace=True)

    return data
