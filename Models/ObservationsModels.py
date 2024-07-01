import numpy as np
from abc import ABC, abstractmethod


class ObservationModel(ABC):
    def __init__(self, H, R):
        self.H = H  # observation matrix
        self.R = R  # measurement noise covariance

    @abstractmethod
    def generate_obs(self, true_state):
        pass


class GaussianMeasModel(ObservationModel):
    def __init__(self, meas_noise_std):
        H = np.array([1., 0]).reshape(1, 2)
        self.meas_noise_std = meas_noise_std
        R = np.array([meas_noise_std ** 2]).reshape(1, 1)
        super().__init__(H, R)

    def generate_obs(self, true_state):
        return self.H @ true_state + np.random.normal(loc=0, scale=self.meas_noise_std)
