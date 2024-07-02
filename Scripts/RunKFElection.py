import matplotlib.pyplot as plt

from Scripts.ElectionType import election_2005, election_2010, election_2015, election_2017, election_2019
from Scripts.ElectionPredictor import KalmanElectionPredictor
from Filters.KalmanFilter import KalmanFilter
from Models.ProcessModels import ConstantVelocityModel, ConstantAccModel
from Models.ObservationsModels import GaussianMeasModel

if __name__ == "__main__":

    obs_model = GaussianMeasModel(5.)
    #process_model = ConstantVelocityModel(0.001)
    process_model = ConstantAccModel(0.1)
    kf = KalmanFilter(observation_model=obs_model, process_model=process_model)

    election_predictor = KalmanElectionPredictor(kf)
    election_predictor.add_elections(*[election_2005, election_2010, election_2015, election_2017, election_2019])

    election_predictor.run()
    plt.show()
