import numpy as np


class ParticleSet:
    def __init__(self, particles, N):
        self.particles = particles
        self.weights = np.ones(N) / N
        self.particle_num = N

    @classmethod
    def create_gaussian_particles(cls, mean: np.array, std: np.array, N: int):
        particles = np.random.multivariate_normal(mean, np.diag(std), size=N)
        #particles[:, 2] %= 2 * np.pi
        return cls(particles, N)

    @classmethod
    def create_uniform_particles(cls, bounds, N):
        dim = len(bounds)
        particles = np.empty((dim, N))
        for idx, bound in enumerate(bounds):
            particles[idx] = np.random.uniform(bound[0], bound[1], size=N)
        #particles[:, 2] %= 2 * np.pi
        return cls(particles, N)

    def get_estimate(self):
        """returns mean and variance of the weighted particles"""

        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=self.weights, axis=0)
        return mean, var

    def normalise_weights(self):
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def get_neff(self):
        return 1. / np.sum(np.square(self.weights))

    def __len__(self):
        return len(self.particles)
