import numpy as np
import matplotlib.pyplot as plt

from Utils.DataStore import DataStore
from Utils.Utils import read_election_data, get_dt_from_election_date
from Scripts.ElectionType import Election


class ElectionPredictor:
    def __init__(self, estimator):
        self.estimator = estimator
        self.elections = []

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

            data_store.plot_political(ax, party=party,
                                      plot_params={'plot_gt': False, 'plot_cov': True, 'plot_meas': False,
                                                   'plot_tracks': True})

            self.predict_election_day(election, time_array[-1], data_store, party)

            ax.scatter([get_dt_from_election_date(election.election_date, 0)],
                       [election.result[party]], marker='x', color=data_store.party_col_map[party])

        print(election.result)

    def predict_election_day(self, election: Election, most_recent_poll_ts: float, data_store, party: str):
        dt = get_dt_from_election_date(election.election_date, most_recent_poll_ts)
        x, P = self.estimator.predict(*data_store.get_most_recent_estimate(), dt)
        print(f'{party}: {np.round(x[0].item(0), 2)} +-', f'{np.round(np.sqrt(P[0, 0]), 2)}')

    def add_elections(self, *elections):
        for election in elections:
            self.elections.append(election)

    def run(self):
        for election in self.elections:
            self.predict_election(election)
