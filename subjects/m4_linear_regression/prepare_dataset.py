from dataset_man import dataset_manager
import numpy as np

class DataDataset():
    def __init__(self) -> None:
        self.__dataset_id = 'm4.vectorization_data'
        self.__X, self.__y = self.__process_dataset()

    @property
    def X(self):
        return self.__X
    
    @property
    def y(self):
        return self.__y

    def output(self) -> tuple:
        return (self.__X, self.__y)
    
    def __process_dataset(self):
        data_path = dataset_manager.get_dataset_path_by_id(self.__dataset_id)
        data = np.genfromtxt(data_path, delimiter=',').tolist()
        
        x_data  = self.__get_column(data, 0)
        y_data = self.__get_column(data, 1)

        return x_data, y_data

    def __get_column(self, data, index):
        result = [row[index] for row in data]

        return result
