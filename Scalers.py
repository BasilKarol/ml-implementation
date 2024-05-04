import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class AbsStandardizer(ABC):
    '''
     - data_to_stadnardize: dict[str : np.array]
        Dictionary with keys as data names (headers) and values as numpy matrix, 
    representing data. 
     - data: dict[str : np.array]:
        Standardized data is stored here.
    '''
    
    def __init__(self, **data_to_stadnardize):
        self.data_to_stadnardize = data_to_stadnardize
        self.data = {}
        self.init_normalization()    
    
    def __getitem__(self, key: str) -> np.array:
        return self.data[key]
    
    @abstractmethod
    def init_normalization(self):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def denormalize(self):
        pass


class MinMaxStandardizer(AbsStandardizer):

    def init_normalization(self):
        self._params = {}
        for key, value in self.data_to_stadnardize.items():
            data_min, data_max = value.min(axis=0), value.max(axis=0)
            # Storing the Standarsizer main parameters into dictionary:
            self._params.update(
               {
                   key : {
                        "min" : data_min,
                        "max" : data_max
                        }
                }
            )
            self.data[key] = (value - data_min) / (data_max - data_min)
    
    def normalize(self, data: np.array, key: str) -> np.array:
        return (data - self._params[key]["min"]) \
                / (self._params[key]["max"] - self._params[key]["min"])
    
    def denormalize(self, data: np.array, key: str) -> np.array:
        return data * (self._params[key]["max"] - self._params[key]["min"]) \
            + self._params[key]["min"]


class NormalStandardizer(AbsStandardizer):

    def init_normalization(self):
        self._params = {}
        for key, value in self.data_to_stadnardize.items():
            data_mean, data_std = value.mean(axis=0), value.std(axis=0)
            # Storing the Standarsizer main parameters into dictionary:
            self._params.update(
               {
                   key : {
                        "mean" : data_mean,
                        "std" : data_std
                        }
                }
            )
            self.data[key] = (value - data_mean) / data_std
        
    def normalize(self, data: np.array, key: str) -> np.array:
        return (data - self._params[key]["mean"]) / self._params[key]["std"]
    
    def denormalize(self, data: np.array, key: str) -> np.array:
        return data * self._params[key]["std"] + self._params[key]["mean"]


class RobustStandardizer(AbsStandardizer):

    def init_normalization(self):
        self._params = {}
        for key, value in self.data_to_stadnardize.items():
            data_q1, data_q3 = np.quantile(value, 0.25, axis=0), np.quantile(value, 0.75, axis=0)
            data_median = np.median(value, axis=0)
            # Storing the Standarsizer main parameters into dictionary:
            self._params.update(
                {
                    key : {
                        "q1" : data_q1,
                        "median" : data_median,
                        "q3" : data_q3
                        }
                }
            )
            self.data[key] = (value - data_median) / (data_q3 - data_q1)
        
    def normalize(self, data: np.array, key: str) -> np.array:
        return (data - self._params[key]["median"]) / ( self._params[key]["q3"] - self._params[key]["q1"] )
    
    def denormalize(self, data: np.array, key: str) -> np.array:
        return data * (self._params[key]["q3"] - self._params[key]["q1"]) \
            + self._params[key]["median"]
    

class NoneStandardizer(AbsStandardizer):

    def init_normalization(self):
        self._params = {}

    def normalize(self, *args) -> Any:
        return args
    
    def denormalize(self, *args) -> Any:
        return args
    
    