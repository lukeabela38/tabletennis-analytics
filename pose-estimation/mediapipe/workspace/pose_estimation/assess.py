import numpy as np
from tslearn.metrics import dtw as ts_dtw
from scipy import stats

class PoseEstimationAssessor():
    def __init__(self, enable_euclidean: bool = False, 
                 enable_mape: bool = False, 
                 enable_correlation: bool = True, 
                 enable_dtw: bool = False,
                 normalize: bool = False
                 ):

        self.normalize = normalize
        self.enable_euclidean = enable_euclidean
        self.enable_mape = enable_mape
        self.enable_correlation = enable_correlation
        self.enable_dtw = enable_dtw
    
    def classify(self, actual, pred):
        
        dict = {}
        if self.normalize == True:
            actual = stats.zscore(actual)
            pred = stats.zscore(pred)

        if self.enable_euclidean:
            dict["euclidean"] = self.euclidean(actual, pred)
        if self.enable_mape:
            dict["mape"] = self.mape(actual, pred)
        if self.enable_correlation:
            dict["correlation"] = self.correlation(actual, pred)
        if self.enable_dtw:
            dict["dtw"] = self.dtw(actual, pred)

        return dict

    @staticmethod
    def euclidean(actual, pred):
        assert len(actual) == len(pred)
        return np.sqrt(np.sum((actual - pred) ** 2))

    @staticmethod
    ## mean average percentage error
    def mape(actual, pred):
        assert len(actual) == len(pred)
        return np.mean(np.abs((actual - pred) / actual))
    
    @staticmethod
    def correlation(actual, pred):
        assert len(actual) == len(pred)
        a_diff = actual - np.mean(actual)
        p_diff = pred - np.mean(pred)
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
        return numerator / denominator
    
    @staticmethod
    def dtw(actual, pred):
        return ts_dtw(actual, pred)