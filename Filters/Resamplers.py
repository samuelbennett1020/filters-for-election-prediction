import numpy as np
from abc import ABC, abstractmethod

from KF_Election.Filters.ParticleSet import ParticleSet


class Resampler(ABC):

    @classmethod
    def resample_from_index(cls, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights.resize(len(particles))
        weights.fill(1.0 / len(weights))

    @classmethod
    @abstractmethod
    def _resample_method(cls, weights: np.array):
        pass

    @classmethod
    def resample(cls, particle_set: ParticleSet):
        indexes = cls._resample_method(particle_set.weights)
        cls.resample_from_index(particle_set.particles, particle_set.weights, indexes)


class MultinomalResampler(Resampler):

    @classmethod
    def _resample_method(cls, weights: np.array) -> np.array:
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # avoid round-off errors
        return np.searchsorted(cumulative_sum, np.random.random(len(weights)))


class ResidualResampler(Resampler):

    @classmethod
    def _resample_method(cls, weights: np.array) -> np.array:
        N = len(weights)
        indexes = np.zeros(N, 'i')

        # take int(N*w) copies of each weight
        num_copies = (N*np.asarray(weights)).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1

        # use multinormial resample on the residual to fill up the rest.
        residual = weights - num_copies     # get fractional part
        residual /= sum(residual)     # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.  # ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))

        return indexes


class StratifiedResampler(Resampler):

    @classmethod
    def _resample_method(cls, weights: np.array) -> np.array:
        N = len(weights)
        # make N subdivisions, chose a random position within each one
        positions = (np.random.random(N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes


class SystematicResampler(Resampler):

    @classmethod
    def _resample_method(cls, weights: np.array) -> np.array:
        N = len(weights)

        # make N subdivisions, choose positions
        # with a consistent random offset
        positions = (np.arange(N) + np.random.random()) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes


class SimpleResampler(Resampler):

    @classmethod
    def _resample_method(cls, weights: np.array) -> np.array:
        N = len(weights)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        return indexes
