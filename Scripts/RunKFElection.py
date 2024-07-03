import matplotlib.pyplot as plt

from ElectionPredictor.ElectionType import election_2005, election_2010, election_2015, election_2017, election_2019, election_2024
from ElectionPredictor.ElectionPredictor import KalmanElectionPredictor
from Filters.KalmanFilter import KalmanFilter
from Models.ProcessModels import ConstantAccModel, ConstantVelocityModel
from Models.ObservationsModels import GaussianMeasModel

if __name__ == "__main__":

    KalmanElectionPredictor.stand_devs = 2.

    obs_model = GaussianMeasModel(8.)
    process_model = ConstantVelocityModel(0.001)
    #process_model = ConstantAccModel(0.1)
    kf = KalmanFilter(observation_model=obs_model, process_model=process_model)

    election_predictor = KalmanElectionPredictor(kf, log_to_file=True)
    election_predictor.add_elections(*[election_2005, election_2010, election_2015, election_2017, election_2019, election_2024])

    election_predictor.run()
    election_predictor.stop_log()
    plt.show()
    plt.legend()
