import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from Utils.DataStore import DataStore
from Filters.KalmanFilter import KalmanFilter
from Models.ProcessModels import ConstantVelocityModel
from Models.ObservationsModels import GaussianMeasModel
from Utils.Utils import read_election_data, get_dt_from_election_date


if __name__ == '__main__':

    year = '2017-19'

    fig, ax = plt.subplots()
    for party in ['Lab', 'Con', 'LD']:#, 'BXP/Reform', 'Green']:
        time_array, party_data = read_election_data(party, year=year)
        LARGE = 1.
        x, P = np.array([[party_data[0]], [0.]]), np.array([[LARGE, 0.], [0., LARGE]])

        obs_model = GaussianMeasModel(8.)
        process_model = ConstantVelocityModel(0.0001)

        kf = KalmanFilter(observation_model=obs_model, process_model=process_model)
        data_store = DataStore()

        for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
            dt = time_step - time_array[idx]
            x, P = kf.predict(x, P, dt)
            x, P = kf.update(x, P, data)
            data_store.add(None, np.array(data).reshape(1, 1), x, P, time_step)

        data_store.plot_political(ax, party=party,
                                  plot_params={'plot_gt': False, 'plot_cov': True, 'plot_meas': True, 'plot_tracks': True})

        dt = get_dt_from_election_date(datetime(2019, 12, 12), time_array[-1])
        x, P = kf.predict(*data_store.get_most_recent_estimate(), dt)
        print(f'{party}: {np.round(x[0].item(0), 2)} +-', f'{np.round(np.sqrt(P[0, 0]), 2)}')

    print('Real: Con: 43.6, Lab: 32.2, LD: 11.5')

    plt.legend()
    plt.show()
