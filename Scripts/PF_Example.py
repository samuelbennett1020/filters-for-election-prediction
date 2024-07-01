import matplotlib.pyplot as plt
import numpy as np

from KF_Election.Filters.ParticleSet import ParticleSet
from KF_Election.Filters.SIRFilter import SIRParticleFilter
from KF_Election.Models.ProcessModels import ConstantVelocityModel
from KF_Election.Models.ObservationsModels import GaussianMeasModel
from KF_Election.DataStore import ParticleDataStore


def run_example_pf(N, iters=5):
    true_state = np.array([[0.5], [0.5]])

    sensor_std_error = 0.8
    q = 0.005
    process_model = ConstantVelocityModel(q)
    obs_model = GaussianMeasModel(sensor_std_error)
    pf = SIRParticleFilter()

    particle_set = ParticleSet.create_gaussian_particles(true_state.flatten(), np.array([sensor_std_error, 0.5]), N)
    data_store = ParticleDataStore()

    t = 0
    dt = 1.
    for x in range(iters):
        t += dt
        true_state = process_model.process(true_state, dt)
        meas = obs_model.generate_obs(true_state)

        pf.predict(particle_set.particles, dt, q)
        pf.update(particle_set, z=meas, R=sensor_std_error)

        if particle_set.get_neff() < N / 2:  # resample if too few effective particles
            pf.resample(particle_set)

        data_store.add(particle_set, true_state, meas, None, None, t)

    mu, var = particle_set.get_estimate()
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    return data_store


if __name__ == "__main__":
    fig, ax = plt.subplots()
    np.random.seed(2)

    data_store = run_example_pf(N=1000, iters=20)
    data_store.plot(plot_params={'plot_gt': True, 'plot_cov': False, 'plot_tracks': False, 'plot_meas': True}, ax=ax)

    plt.legend()
    plt.show()
