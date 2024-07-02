import matplotlib.pyplot as plt

from Scripts.ElectionType import election_2005, election_2010, election_2015, election_2017, election_2019
from Scripts.ElectionPredictor import PFElectionPredictor
from Filters.SIRFilter import SIRParticleFilter
from Filters.Resamplers import StratifiedResampler
from Models.ProcessModels import ConstantVelocityModel
from Models.ObservationsModels import GaussianMeasModel

if __name__ == "__main__":

    obs_model = GaussianMeasModel(4.)
    process_model = ConstantVelocityModel(0.001)
    pf = SIRParticleFilter(observation_model=obs_model, process_model=process_model, resampler=StratifiedResampler())

    election_predictor = PFElectionPredictor(pf, num_particles=5_000)
    election_predictor.add_elections(*[election_2005, election_2010, election_2015, election_2017, election_2019])

    election_predictor.run()
    plt.show()
