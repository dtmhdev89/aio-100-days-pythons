import numpy as np
import pandas as pd

from dataset_man import dataset_manager

class AdvertisingDataset():
    def __init__(self) -> None:
        self.dataset_id = 'm4.advertising'
        self.__X, self.__y = self.__load_dataset()

    @property
    def X(self):
        return self.__X
    
    @property
    def y(self):
        return self.__y

    def __load_dataset(self) -> tuple:
        dataset_path = dataset_manager.get_dataset_path_by_id(self.dataset_id)
        data = np.genfromtxt(dataset_path, delimiter=',', skip_header=1)
        X = data[:, :3]
        y = data[:, 3]

        return X, y
    
class BTCDataset():
    def __init__(self) -> None:
        self.__id = 'm4.BTC_Daily'
        self.__df = self.__read_and_preprocess_dataset()
    
    @property
    def df(self):
        return self.__df
    
    def __read_and_preprocess_dataset(self):
        df = dataset_manager.load_dataset(self.__id, as_dataframe=True)
        df = df.drop_duplicates()

        return df
