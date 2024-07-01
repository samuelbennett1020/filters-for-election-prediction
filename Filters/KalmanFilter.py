import numpy as np

from Utils.Utils import inv
from Models.ProcessModels import ProcessModel
from Models.ObservationsModels import ObservationModel


class KalmanFilter:
    def __init__(self, process_model: ProcessModel, observation_model: ObservationModel):
        self.process_model = process_model
        self.observation_model = observation_model

    def predict(self, x, P, dt):
        F = self.process_model.F(dt)
        x_predict = F @ x
        P_predict = F @ P @ F.T + self.process_model.Q(dt)
        return x_predict, P_predict

    def update(self, x, P, z):
        H = self.observation_model.H
        y = z - H @ x  # innovation
        S = H @ P @ H.T + self.observation_model.R  # innovation covariance
        K = P @ H.T @ inv(S)  # kalman gain
        x_update = x + K @ y
        P_update = (np.eye(len(x)) - K @ H) @ P
        return x_update, P_update
