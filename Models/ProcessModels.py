import numpy as np
from abc import ABC, abstractmethod


class ProcessModel(ABC):
    def __init__(self, F, Q):
        self.F = F  # transition matrix
        self.Q = Q  # process noise covariance

    @abstractmethod
    def process(self, true_state, dt):
        pass


class ConstantVelocityModel(ProcessModel):
    def __init__(self, q):
        F = lambda t: np.array([[1., t], [0., 1.]])
        Q = lambda t: np.array([[(t**3)/3, (t**2)/2], [(t**2)/2, t]])*q
        super().__init__(F, Q)

    def process(self, true_state, dt):
        return self.F(dt) @ true_state + np.random.multivariate_normal(np.array([0., 0.]), cov=self.Q(dt)).reshape(2, 1)


class ConstantAccModel(ProcessModel):
    def __init__(self, process_noise_std):
        F = lambda t: np.array([[1., t], [0., 1.]])

        self.G = lambda t: np.array([[0.5 * t ** 2], [t]])
        self.process_noise_std = process_noise_std

        Q = lambda t: self.G(t) @ self.G(t).T * self.process_noise_std ** 2

        super().__init__(F, Q)

    def process(self, true_state, dt):
        return self.F(dt) @ true_state + self.G(dt) * np.random.normal(loc=0, scale=self.process_noise_std)
