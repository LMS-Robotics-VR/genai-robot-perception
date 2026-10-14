from ultralytics import YOLO
from typing import Union


class Trainer:
    def __init__(self,
                 model_weights):
        self.model_weights = model_weights
        self.model = YOLO(model_weights)

    def train(self,
              data: str,
              epochs: int,
              imgsz: int = 640,
              batch: int = 16,
              save: bool = True,
              device: Union[int,list,str] = "cpu",
              lr0: float = 0.01,
              lrf: float = 0.1,
              half: bool = False)->None:
        
        '''
        Trains a model

        -Args:
            data (str): path for data.yaml configuration file
            epochs (int): number of epochs
            imgsz (int): image width which model will be trained
            batch (int): batch size
            save (bool): save for results of training
            device (int | list | str): number of GPU | list of numbers | "cpu"
            lr0 (float): initial learning rate in first epoch
            lrf (float): calculates final learning rate (lr0*lrf)
            amp (bool): use automatic mixel precision. Needs Nvidia GPU that supports amp
        -Returns:
            (None)
        '''
        self.model.train(data = data,
                         epochs = epochs,
                         imgsz = imgsz,
                         batch = batch,
                         save = save,
                         device = device,
                         lr0 = lr0,
                         lrf = lrf,
                         amp = False)