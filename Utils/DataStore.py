from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from Filters.ParticleSet import ParticleSet

# COLOURS
RED = np.array([255, 0, 0]) / 255
BLUE = np.array([0, 0, 255]) / 255
ORANGE = np.array([255, 140, 0]) / 255
PURPLE = np.array([128, 0, 128])/255
GREEN = np.array([0, 255, 0]) / 255
LIGHT_BLUE = np.array([0, 255, 255]) / 255


class DataStore:

    party_col_map = {'Lab': RED, 'Con': BLUE, 'LD': ORANGE, 'Ukip': PURPLE, 'Green': GREEN, 'BXP/Reform': LIGHT_BLUE}

    def __init__(self, label: str = None):
        self.gts = []
        self.states = []
        self.covs = []
        self.times = []
        self.measurements = []
        self.label = label

    def add(self, gt, meas, state, cov, time):
        self.gts.append(deepcopy(gt))
        self.states.append(deepcopy(state))
        self.covs.append(deepcopy(cov))
        self.times.append(deepcopy(time))
        self.measurements.append(deepcopy(meas))

    def plot_political(self, ax: tuple = None,
                       plot_params: dict = {'plot_gt': False, 'plot_cov': True, 'plot_meas': False},
                       party: str = None):

        color = self.party_col_map[party]
        if not ax:
            _, ax = plt.subplots()

        if plot_params['plot_gt']:
            ax.plot_date(self.times, [x[0] for x in self.gts], label='Ground Truth', fmt='-', color=color)

        tracks = np.array([x[0] for x in self.states]).flatten()
        ax.plot_date(self.times, tracks, label=party, fmt='-', color=color)

        if plot_params['plot_cov']:
            covs = np.array([np.sqrt(x[0][0]) for x in self.covs])
            ax.fill_between(self.times, tracks + covs, tracks - covs, alpha=0.4, color=color)

        if plot_params['plot_meas']:
            ax.scatter(self.times, [x[0] for x in self.measurements], color=color)

        ax.set_ylim(0)

        ax.set_ylabel('Vote Share')

    def plot(self, ax=None,
             plot_params: dict = {'plot_gt': False, 'plot_cov': True, 'plot_tracks': True, 'plot_meas': True}):

        if not ax:
            _, ax = plt.subplots()

        if plot_params['plot_gt']:
            ax.plot(self.times, [x[0] for x in self.gts], label='Ground Truth', color=BLUE)

        if plot_params['plot_tracks']:
            tracks = np.array([x[0] for x in self.states]).flatten()
            ax.plot(self.times, tracks, label='Track', color=GREEN)

        if plot_params['plot_cov']:
            covs = np.array([np.sqrt(x[0][0]) for x in self.covs])
            ax.fill_between(self.times, tracks + covs, tracks - covs, alpha=0.4, color=GREEN)

        if plot_params['plot_meas']:
            ax.scatter(self.times, [x[0] for x in self.measurements], color=RED, label='Measurements', marker='x')

    def get_most_recent_estimate(self) -> (np.array, np.array):
        return deepcopy(self.states[-1]), deepcopy(self.covs[-1])


class ParticleDataStore(DataStore):

    party_to_mpl_colormap = {'Lab': 'Reds', 'Con': 'Blues', 'LD': 'Oranges',
                             'Ukip': 'Purples', 'Green': 'Greens', 'BXP/Reform': 'PuBu'}

    def __init__(self, label: str = None):
        self.particles: [ParticleSet] = []
        super().__init__(label)

    def add(self, particles: ParticleSet, gt, meas, state, cov, time):
        super().add(gt, meas, state, cov, time)
        self.particles.append(deepcopy(particles))

    def plot(self, ax=None,
             plot_params: dict = {'plot_gt': False, 'plot_cov': True, 'plot_tracks': True, 'plot_meas': True}):
        super().plot(ax, plot_params)

        for particle_set, time in zip(self.particles, self.times):
            mu, _ = particle_set.get_estimate()
            ax.scatter([time]*len(particle_set), particle_set.particles[:, 0], alpha=0.01)
            ax.scatter([time], mu[0], marker='s', color='r')


    def plot_2d(self, timestep: int = 0, ax=None, faint: bool = False):
        if ax is None:
            fig, ax = plt.subplots()
        particles_set = self.particles[timestep]
        mu, _ = particles_set.get_estimate()
        if not faint:
            ax.scatter(particles_set.particles[:, 0], particles_set.particles[:, 1], color='k', marker=',', s=1)
            ax.scatter(mu[0], mu[1], marker='s', color='r')
        else:
            alpha = .20
            num_particles = len(particles_set)
            if num_particles > 5000:
                alpha *= np.sqrt(5000) / np.sqrt(num_particles)
            plt.scatter(particles_set.particles[:, 0], particles_set.particles[:, 1], alpha=alpha, color='g')

    def plot_political(self, ax: tuple = None,
                       plot_params: dict = {'plot_gt': False, 'plot_cov': True, 'plot_meas': False},
                       party: str = None):

        color = self.party_col_map[party]
        if not ax:
            _, ax = plt.subplots()

        means, vars = np.array([]), np.array([])
        for particle_set in self.particles:
            mean, var = particle_set.get_estimate()
            means = np.append(means, mean[0])
            vars = np.append(vars, np.sqrt(var[0]))

        ax.plot_date(self.times, means, label=party, fmt='-', color=color)

        if plot_params['plot_cov']:
            ax.fill_between(self.times, means + vars, means - vars, alpha=0.4, color=color)

        if plot_params['plot_meas']:
            ax.scatter(self.times, [x[0] for x in self.measurements], color=color)

        ax.set_ylim(0)

        ax.set_ylabel('Vote Share')

    def plot_political_with_density(self, ax, party: str = None):

        x_data, y_data, weights = np.array([]), np.array([]), np.array([])
        for particle_set, time in zip(self.particles, self.times):
            x_data = np.append(x_data, np.array([time]*len(particle_set)))
            y_data = np.append(y_data, particle_set.particles[:, 0])
            weights = np.append(weights, particle_set.weights)
        ax.hist2d(x_data, y_data, weights=weights, cmap=self.party_to_mpl_colormap[party], cmin=10/len(self.particles[0]),
                  bins=[np.array(self.times), np.linspace(0, 60, 1_000)])

    def get_most_recent_particle_set(self) -> np.array:
        return deepcopy(self.particles[-1])
