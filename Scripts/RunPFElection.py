import matplotlib.pyplot as plt

from ElectionPredictor.ElectionType import election_2005, election_2010, election_2015, election_2017, election_2019, election_2024
from ElectionPredictor.ElectionPredictor import PFElectionPredictor
from Filters.SIRFilter import SIRParticleFilter
from Filters.Resamplers import StratifiedResampler
from Models.ProcessModels import ConstantVelocityModel
from Models.ObservationsModels import GaussianMeasModel


if __name__ == "__main__":

    PFElectionPredictor.stand_devs = 2.

    obs_model = GaussianMeasModel(3.)
    process_model = ConstantVelocityModel(0.001)
    pf = SIRParticleFilter(observation_model=obs_model, process_model=process_model, resampler=StratifiedResampler())

    election_predictor = PFElectionPredictor(pf, num_particles=10_000, log_to_file=True)
    election_predictor.add_elections(*[election_2005, election_2010, election_2015, election_2017, election_2019, election_2024])

    election_predictor.run()
    election_predictor.stop_log()
    plt.legend()
    plt.show()
