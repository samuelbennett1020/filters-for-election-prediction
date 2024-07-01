import numpy as np
import matplotlib.pyplot as plt

from KF_Election.DataStore import DataStore
from KF_Election.Filters.KalmanFilter import KalmanFilter
from KF_Election.Models.ProcessModels import ConstantVelocityModel
from KF_Election.Models.ObservationsModels import GaussianMeasModel
from KF_Election.Utils import read_election_data

if __name__ == '__main__':
    fig, ax = plt.subplots()
    for party in ['Lab', 'Con', 'LD', 'BXP/Reform', 'Green']:
        time_array, party_data = read_election_data(party)
        LARGE = 1.
        x, P = np.array([[party_data[0]], [0.]]), np.array([[LARGE, 0.], [0., LARGE]])

        obs_model = GaussianMeasModel(8.)
        process_model = ConstantVelocityModel(0.001)

        kf = KalmanFilter(observation_model=obs_model, process_model=process_model)
        data_store = DataStore()

        for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
            dt = time_step - time_array[idx]
            x, P = kf.predict(x, P, dt)
            x, P = kf.update(x, P, data)
            data_store.add(None, np.array(data).reshape(1, 1), x, P, time_step)

        data_store.plot_political(ax, party=party,
                                  plot_params={'plot_gt': False, 'plot_cov': True, 'plot_meas': False, 'plot_tracks': True})
    plt.legend()
    plt.show()

