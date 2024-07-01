import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from Filters.ParticleSet import ParticleSet
from Filters.SIRFilter import SIRParticleFilter
from Utils.DataStore import ParticleDataStore
from Utils.Utils import read_election_data, get_dt_from_election_date
from Filters.Resamplers import StratifiedResampler


def run_example_pf(N, party, year):

    sensor_std_error = 8.
    q = 0.003

    time_array, party_data = read_election_data(party, year=year)
    pf = SIRParticleFilter(StratifiedResampler())

    particle_set = ParticleSet.create_gaussian_particles(np.array([party_data[0], 0.]),
                                                         np.array([sensor_std_error, 10.]), N)
    data_store = ParticleDataStore()

    for idx, (time_step, data) in enumerate(zip(time_array[1:], party_data[1:])):
        dt = time_step - time_array[idx]

        pf.predict(particle_set.particles, dt, q)
        pf.update(particle_set, z=data, R=sensor_std_error)

        if particle_set.get_neff() < N / 2:  # resample if too few effective particles
            pf.resample(particle_set)

        data_store.add(particle_set, None, np.array(data).reshape(1, 1), None, None, time_step)

    # July 4 2024 Election Prediction
    dt = get_dt_from_election_date(datetime(2019, 12, 12), time_array[-1])
    particle_set = data_store.get_most_recent_particle_set()
    pf.predict(particle_set.particles, dt, q)
    x, P = particle_set.get_estimate()
    print(f'{party}: {np.round(x[0].item(0), 2)} +-', f'{np.round(np.sqrt(P[0]), 2)}')

    return data_store


if __name__ == "__main__":
    fig, ax = plt.subplots()
    np.random.seed(2)
    plot_params = {'plot_gt': False, 'plot_cov': True, 'plot_tracks': False, 'plot_meas': False}
    for party in ['Lab', 'Con', 'LD']:
        data_store = run_example_pf(N=1_000, party=party, year='2017-19')
        data_store.plot_political(plot_params=plot_params, ax=ax, party=party)
        #data_store.plot_political_with_density(ax, party)

    print('Real: Con: 43.6, Lab: 32.2, LD: 11.5')

    plt.legend()
    plt.show()
