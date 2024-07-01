import scipy.stats
import numpy as np

from KF_Election.Filters.ParticleSet import ParticleSet


class SIRParticleFilter_Robot:

    def predict(self, particles, u, std, dt=1.):
        """ move according to control input u (heading change, velocity)
        with noise Q (std heading change, std velocity)`"""

        N = len(particles)
        # update heading
        particles[:, 2] += u[0] + (np.random.randn(N) * std[0])
        particles[:, 2] %= 2 * np.pi

        # move in the (noisy) commanded direction
        dist = (u[1] * dt) + (np.random.randn(N) * std[1])
        particles[:, 0] += np.cos(particles[:, 2]) * dist
        particles[:, 1] += np.sin(particles[:, 2]) * dist

    def update(self, particle_set: ParticleSet, z, R, landmarks):
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particle_set.particles[:, 0:2] - landmark, axis=1)
            particle_set.weights *= scipy.stats.norm(distance, R).pdf(z[i])

        particle_set.weights += 1.e-300  # avoid round-off to zero
        particle_set.weights /= sum(particle_set.weights)  # normalize

    def simple_resample(self, particle_set: ParticleSet):
        N = len(particle_set.particles)
        cumulative_sum = np.cumsum(particle_set.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        # resample according to indexes
        particle_set.particles[:] = particle_set.particles[indexes]
        particle_set.weights.fill(1.0 / N)

        assert np.allclose(particle_set.weights, 1 / N)
