import scipy.stats
import numpy as np

from Filters.ParticleSet import ParticleSet
from Filters.Resamplers import Resampler
from Models.ObservationsModels import ObservationModel
from Models.ProcessModels import ProcessModel


class SIRParticleFilter:
    # Sequential Importance Resampling Filter

    def __init__(self, resampler: Resampler, observation_model: ObservationModel, process_model: ProcessModel):
        self.resampler = resampler
        self.process_model = process_model
        self.observation_model = observation_model

    def predict(self, particles, dt):
        """ move according to control input u (heading change, velocity)
        with noise Q (std heading change, std velocity)`"""

        N = len(particles)
        Q = lambda t: np.array([[(t ** 3) / 3, (t ** 2) / 2], [(t ** 2) / 2, t]]) * self.process_model.q  # TODO: refactor
        acc_term = np.random.multivariate_normal(np.array([0., 0.]), cov=Q(dt), size=N)
        particles[:, 0] += particles[:, 1]*dt + acc_term[:, 0]
        particles[:, 1] += acc_term[:, 1]

    def update(self, particle_set: ParticleSet, z):
        coeff = scipy.stats.norm(particle_set.particles[:, 0], self.observation_model.R).pdf(z).flatten()  # importance function
        particle_set.weights *= coeff

        particle_set.normalise_weights()

    def resample(self, particle_set: ParticleSet):
        self.resampler.resample(particle_set)
