import scipy.stats
import numpy as np

from KF_Election.Filters.ParticleSet import ParticleSet
from KF_Election.Filters.Resamplers import Resampler


class SIRParticleFilter:
    # Sequential Importance Resampling Filter

    def __init__(self, resampler: Resampler):
        self.resampler = resampler

    def predict(self, particles, dt, q):
        """ move according to control input u (heading change, velocity)
        with noise Q (std heading change, std velocity)`"""

        N = len(particles)
        Q = lambda t: np.array([[(t ** 3) / 3, (t ** 2) / 2], [(t ** 2) / 2, t]]) * q
        acc_term = np.random.multivariate_normal(np.array([0., 0.]), cov=Q(dt), size=N)
        particles[:, 0] += particles[:, 1]*dt + acc_term[:, 0]
        particles[:, 1] += acc_term[:, 1]

    def update(self, particle_set: ParticleSet, z, R):
        coeff = scipy.stats.norm(particle_set.particles[:, 0], R).pdf(z).flatten()  # importance function
        particle_set.weights *= coeff

        particle_set.normalise_weights()

    def resample(self, particle_set: ParticleSet):
        self.resampler.resample(particle_set)
