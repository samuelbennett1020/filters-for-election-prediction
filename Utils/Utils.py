import numpy as np
import pandas as pd
import time
from datetime import datetime, date, timedelta
import os


pd.options.mode.chained_assignment = None  # default='warn'


def inv(a: np.array) -> np.array:
    try:
        return np.linalg.inv(a)
    except:
        return (1/a).reshape(1, 1)


def get_date_from_days_after(start: date, end: int):
    return start + timedelta(end)


def get_dt_from_election_date(election_date, most_recent_ts: float):
    return election_date.timestamp() / (60 * 60 * 24) - most_recent_ts


def get_time_array(df):
    time_array = np.array([])

    for date_ in df['Unnamed: 3']:
        index = date_.rfind('/')
        new_date = date_[:index + 1] + '20' + date_[index + 1:]

        new_date = datetime.strptime(new_date, "%d/%m/%Y")
        if new_date.date() > datetime.today().date():  # sanity check
            continue

        time_array = np.append(time_array, time.mktime(new_date.timetuple()))

    #time_array -= time_array[0]
    time_array /= 60 * 60 * 24  # convert to days
    return time_array


def read_election_data(party: str, year: str = None):
    dirname = os.path.dirname(__file__)

    if year is None:
        fp = '../Data/NationalVotingIntention2019+.csv'
    else:
        fp = f'../Data/NationalVotingIntention{year}.csv'

    filename = os.path.join(dirname, fp)

    with open(filename, encoding='unicode_escape') as f:
        df = pd.DataFrame(pd.read_csv(f))

    df_reduced = df[[party, 'Unnamed: 3']]
    df_reduced.loc[df_reduced[party] == ' '] = np.nan
    df_reduced = df_reduced.dropna()
    party_data = np.array(df_reduced[party])
    time_array = get_time_array(df_reduced)

    # ensure sorted
    ind = np.argsort(time_array)
    time_array = time_array[ind]
    party_data = party_data[ind]

    party_data = party_data.astype(np.float16)

    return time_array, party_data
