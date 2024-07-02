import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from Filters.ParticleSet import ParticleSet
from Utils.DataStore import ParticleDataStore
from Utils.Utils import read_election_data, get_dt_from_election_date
from Utils.DataStore import DataStore
from Scripts.ElectionType import Election


class KalmanElectionPredictor:
    def __init__(self, estimator):
        self.estimator = estimator
        self.elections = []
        self.plot_params = {'plot_gt': False, 'plot_cov': True, 'plot_meas': False, 'plot_tracks': True}

    def predict_election(self, election: Election):
        fig, ax = plt.subplots()

        print(f'----------------------------Election: {election.election_date.date()}---------------------------------')

        for party in election.parties:
            time_array, party_data = read_election_data(party, year=election.year)
            LARGE = 1.
            x, P = np.array([[party_data[0]], [0.]]), np.array([[LARGE, 0.], [0., LARGE]])
            data_store = DataStore()

            for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
                dt = time_step - time_array[idx]
                x, P = self.estimator.predict(x, P, dt)
                x, P = self.estimator.update(x, P, data)
                data_store.add(None, np.array(data).reshape(1, 1), x, P, time_step)

            data_store.plot_political(ax, party=party, plot_params=self.plot_params)

            self.predict_election_day(election, time_array[-1], data_store, party)

            data_store.plot_result(ax, election, party)

    def predict_election_day(self, election: Election, most_recent_poll_ts: float, data_store, party: str):
        dt = get_dt_from_election_date(election.election_date, most_recent_poll_ts)
        x, P = self.estimator.predict(*data_store.get_most_recent_estimate(), dt)
        print(f'{party}: Predicted: {np.round(x[0].item(0), 2)} +-',
              f'{np.round(np.sqrt(P[0, 0]), 2)}; Actual {election.result[party]}')

    def add_elections(self, *elections):
        for election in elections:
            self.elections.append(election)

    def run(self):
        for election in self.elections:
            self.predict_election(election)


class PFElectionPredictor(KalmanElectionPredictor):

    def __init__(self, estimator, num_particles: int):
        super().__init__(estimator)
        self.num_particles = num_particles

    def predict_election(self, election: Election):

        fig, ax = plt.subplots()

        print(f'----------------------------Election: {election.election_date.date()}---------------------------------')

        for party in election.parties:

            time_array, party_data = read_election_data(party, year=election.year)

            particle_set = ParticleSet.create_gaussian_particles(np.array([party_data[0], 0.]),
                                                                 np.array([8., 10.]), self.num_particles)
            data_store = ParticleDataStore()

            for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
                dt = time_step - time_array[idx]

                self.estimator.predict(particle_set.particles, dt)
                self.estimator.update(particle_set, z=data)

                if particle_set.get_neff() < self.num_particles / 2:  # resample if too few effective particles
                    self.estimator.resample(particle_set)

                data_store.add(particle_set, None, np.array(data).reshape(1, 1), None, None, time_step)

            data_store.plot_political(plot_params=self.plot_params, ax=ax, party=party)

            self.predict_election_day(election, time_array[-1], data_store, party)

    def predict_election_day(self, election: Election, most_recent_poll_ts: float, data_store, party: str):
        dt = get_dt_from_election_date(election.election_date, most_recent_poll_ts)
        particle_set = data_store.get_most_recent_particle_set()
        self.estimator.predict(particle_set.particles, dt)
        x, P = particle_set.get_estimate()
        print(f'{party}: {np.round(x[0].item(0), 2)} +-', f'{np.round(np.sqrt(P[0]), 2)}',
              f'Actual {election.result[party]}')
