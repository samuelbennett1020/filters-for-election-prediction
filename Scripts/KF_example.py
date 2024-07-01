import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from KF_Election.DataStore import DataStore
from KF_Election.Filters.KalmanFilter import KalmanFilter
from KF_Election.Models.ProcessModels import ConstantVelocityModel
from KF_Election.Models.ObservationsModels import GaussianMeasModel


if __name__ == "__main__":

    LARGE = 100.
    x, P = np.array([[0.], [0.]]), np.array([[LARGE, 0.], [0., LARGE]])

    obs_model = GaussianMeasModel(5.)
    process_model = ConstantVelocityModel(0.001)

    kf = KalmanFilter(observation_model=obs_model, process_model=process_model)
    data_store = DataStore()

    timesteps = 200
    true_state = deepcopy(x)
    t = 0
    dt = 1.
    for time_step in range(timesteps):
        t += dt
        true_state = process_model.process(true_state, dt)
        x, P = kf.predict(x, P, dt)

        meas = obs_model.generate_obs(true_state)
        x, P = kf.update(x, P, meas)
        data_store.add(true_state, meas, x, P, t)

    data_store.plot(plot_params={'plot_gt': True, 'plot_cov': True, 'plot_meas': True, 'plot_tracks': True})
    plt.legend()
    plt.show()
