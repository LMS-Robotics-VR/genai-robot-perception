import os
import yaml


class YamlModifier:
    def __init__(self,
                 file_path):
        self.file_path = file_path


    def _load_yaml(self)->None:
        '''
        Loads a yaml file from self.file_path attribute

        -Args:
            (None)
        -Returns:
            (None)
        '''

        with open(self.file_path, "r") as file:
            self.data = yaml.load(file, Loader=yaml.Loader)


    def _write_yaml(self)->None:
        '''
        Writes data in yaml file to self.file_path attribute

        -Args:
            (None)
        -Returns:
            (None)
        '''
        
        with open(self.file_path, "w") as file:
            yaml.dump(self.data, file)


    def modify_yaml(self,
                    new_label:str)->None:
        '''
        Loads yaml, modifies it, saves modifications

        -Args:
            new_label (str): new label name to add in dataset labels
        -Returns:
            (None)
        '''
        
        self._load_yaml()
        
        self.data["names"].append(new_label)
        self.data["nc"] += 1
        self.data["test"] = os.path.dirname(self.file_path) + "/test/images"
        self.data["val"] = os.path.dirname(self.file_path) + "/valid/images"
        self.data["train"] = os.path.dirname(self.file_path) + "/train/images"

        self._write_yaml()
