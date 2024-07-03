import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, date

from Filters.ParticleSet import ParticleSet
from Utils.DataStore import ParticleDataStore
from Utils.Utils import read_election_data, get_dt_from_election_date
from Utils.DataStore import DataStore
from ElectionPredictor.ElectionType import Election


class KalmanElectionPredictor:

    stand_devs = 2.

    def __init__(self, estimator, log_to_file: bool = False):
        self.estimator = estimator
        self.elections = []
        self.plot_params = {'plot_gt': False, 'plot_cov': True, 'plot_meas': False, 'plot_tracks': True}

        self.log_file = None
        if log_to_file:
            self.start_log()

        print(f'Error is {self.stand_devs} sigma')

    def start_log(self):

        predictor = 'KF' if self.__class__.__name__ == "KalmanElectionPredictor" else 'PF'

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f'../Results/{predictor}_Log_{date.today()}.txt')
        self.log_file = open(filename, 'w')
        sys.stdout = self.log_file
        print(f'----------------------------Logging Run at: {datetime.now()}---------------------------------')

    def stop_log(self):
        self.log_file.close()

    def predict_election(self, election: Election):
        fig, ax = plt.subplots()
        date = election.election_date.date()
        fig.suptitle(f' Election {date}')

        print(f'----------------------------Election: {date}---------------------------------')

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

            self.predict_election_day(election, time_array[-1], data_store, party, ax)
            data_store.plot_political(ax, party=party, plot_params=self.plot_params)

            data_store.plot_result(ax, election, party)

    def predict_election_day(self, election: Election, most_recent_poll_ts: float, data_store, party: str, ax):
        dt = get_dt_from_election_date(election.election_date, most_recent_poll_ts)
        x, P = self.estimator.predict(*data_store.get_most_recent_estimate(), dt)
        data_store.add(None, None, x, P, election.election_date.timestamp() / (60 * 60 * 24))
        print(f'{party}: {np.round(x[0].item(0), 2)} +-',
              f'{np.round(self.stand_devs*np.sqrt(P[0, 0]), 2)}; Actual {election.result[party]}')
        data_store.plot_prediction(ax, election, party, x[0])

    def add_elections(self, *elections):
        for election in elections:
            self.elections.append(election)

    def run(self):
        for election in self.elections:
            self.predict_election(election)


class PFElectionPredictor(KalmanElectionPredictor):

    def __init__(self, estimator, num_particles: int, log_to_file: bool = False):
        super().__init__(estimator, log_to_file)
        self.num_particles = num_particles

    def predict_election(self, election: Election):

        fig, ax = plt.subplots()
        date = election.election_date.date()
        fig.suptitle(f' Election {date}')

        print(f'----------------------------Election: {date}---------------------------------')

        for party in election.parties:

            time_array, party_data = read_election_data(party, year=election.year)

            sensor_std_error = self.estimator.observation_model.R[0, 0]  # TODO: refactor
            particle_set = ParticleSet.create_gaussian_particles(np.array([party_data[0], 0.]),
                                                                 np.array([sensor_std_error, 10.]), self.num_particles)
            data_store = ParticleDataStore()

            for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
                dt = time_step - time_array[idx]

                self.estimator.predict(particle_set.particles, dt)
                self.estimator.update(particle_set, z=data)

                if particle_set.get_neff() < self.num_particles / 2:  # resample if too few effective particles
                    self.estimator.resample(particle_set)

                data_store.add(particle_set, None, np.array(data).reshape(1, 1), None, None, time_step)

            self.predict_election_day(election, time_array[-1], data_store, party, ax)
            data_store.plot_political(plot_params=self.plot_params, ax=ax, party=party)

            data_store.plot_result(ax, election, party)

    def predict_election_day(self, election: Election, most_recent_poll_ts: float, data_store, party: str, ax):
        dt = get_dt_from_election_date(election.election_date, most_recent_poll_ts)
        particle_set = data_store.get_most_recent_particle_set()
        self.estimator.predict(particle_set.particles, dt)
        data_store.add(particle_set, None, None, None, None, election.election_date.timestamp()/(60*60*24))
        x, P = particle_set.get_estimate()
        print(f'{party}: {np.round(x[0].item(0), 2)} +-', f'{np.round(self.stand_devs*np.sqrt(P[0]), 2)}',
              f'Actual {election.result[party]}')
        data_store.plot_prediction(ax, election, party, x[0])
