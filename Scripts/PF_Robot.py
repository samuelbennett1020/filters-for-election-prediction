import matplotlib.pyplot as plt
import numpy as np

from KF_Election.Filters.ParticleSet import ParticleSet
from KF_Election.Filters.SIRFilter_Robot import SIRParticleFilter_Robot
from KF_Election.DataStore import ParticleDataStore


def run_robot_landmark_pf(N, landmarks, iters, sensor_std_err=.1, initial_x=None):
    pf = SIRParticleFilter_Robot()
    data_store = ParticleDataStore()
    # create particles and weights
    if initial_x is not None:
        particle_set = ParticleSet.create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        particle_set = ParticleSet.create_uniform_particles((0, 20), (0, 20), (0, 6.28), N)

    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = (np.linalg.norm(landmarks - robot_pos, axis=1) + (np.random.randn(len(landmarks)) * sensor_std_err))

        pf.predict(particle_set.particles, u=(0.00, 1.414), std=(.2, .05))  # move diagonally forward to (x+1, x+1)

        pf.update(particle_set, z=zs, R=sensor_std_err, landmarks=landmarks)  # incorporate measurements

        if particle_set.get_neff() < N / 2:  # resample if too few effective particle_set
            pf.simple_resample(particle_set)

        data_store.add(particle_set, robot_pos, zs, None, None, x)
        data_store.plot_2d(x, ax)

    mu, var = particle_set.get_estimate()
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    np.random.seed(2)

    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    iters = 15
    run_robot_landmark_pf(N=5000, iters=iters, landmarks=landmarks, initial_x=(1,1, np.pi/4))
    #
    # for i in range(iters):
    #     data_store.plot_2D(ax)

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.show()
