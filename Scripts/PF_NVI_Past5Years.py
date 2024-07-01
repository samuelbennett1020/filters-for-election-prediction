import matplotlib.pyplot as plt
import numpy as np

from Filters.ParticleSet import ParticleSet
from Filters.SIRFilter import SIRParticleFilter
from Utils.DataStore import ParticleDataStore
from Utils.Utils import read_election_data
from Filters.Resamplers import StratifiedResampler


def run_example_pf(N, party):

    sensor_std_error = 8.
    q = 0.001

    time_array, party_data = read_election_data(party)
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

    return data_store


if __name__ == "__main__":
    fig, ax = plt.subplots()
    np.random.seed(2)
    plot_params = {'plot_gt': False, 'plot_cov': True, 'plot_tracks': False, 'plot_meas': False}
    for party in ['Lab', 'Con', 'LD', 'BXP/Reform', 'Green']:
        data_store = run_example_pf(N=1_000, party=party)
        data_store.plot_political(plot_params=plot_params, ax=ax, party=party)
        #data_store.plot_political_with_density(ax, party)

    plt.legend()
    plt.show()
