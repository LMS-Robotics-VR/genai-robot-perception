import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from groundingdino.util.inference import load_model, load_image, predict, annotate

class GroundingDino:
    list_annotations = []
    list_logits = []
    list_phrases = []

    def __init__(self,
                 config_path,
                 weights_path,
                 images_folder,
                 images_extension,
                 box_threshold,
                 text_threshold,
                 text_prompt,
                 export_path,
                 last_num_label):
        self.CONFIG_PATH = config_path
        self.WEIGHTS_PATH = weights_path
        self.IMAGES_FOLDER = images_folder
        self.IMAGES_EXTENSION = images_extension
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.TEXT_PROMPT = text_prompt
        self.EXPORT_PATH = export_path
        self.LAST_NUM_LABEL = last_num_label
        self.model = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)


    def _load_images(self, image_folder:str, extension:str)->list:
        '''
        Loads the images from a folder path

        -Args: 
            image_folder (str): a folder path which contains the images
            extension (str): extension type of image
        -Returns:
            (list): a list with image paths
        '''

        return [file for file in sorted(glob.glob(image_folder + f"/*.{extension}"))]
    

    def _plot(self, image:np.ndarray)->None:
        '''
        Plots an image with OpenCV window

        -Args:
            image (np.ndarray): an image in numpy array format
        -Returns:
            (None) : plots an image
        '''

        cv2.imshow("GroundingDINO prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _infer(self, text_prompt:str,
            image:str,
            box_threshold:float = 0.35,
            text_threshold:float = 0.25
            )->Tuple[torch.Tensor, torch.Tensor, List[str]]:
        '''
        Run inference to get results from the model

        -Args:
            text_prompt (str): the text prompt that will be passed in GroundingDINO
            image (str): image path to predict detections
        -Returns:
            boxes (torch.Tensor): coordinates for detected objects
            logits (torch.Tensor): probabilities for detected objects
            phrases (list): labels for detected objects
        '''

        image_source, image = load_image(image_path=image)

        boxes, logits, phrases = predict(model=self.model,
                                        image=image,
                                        caption=text_prompt,
                                        box_threshold=box_threshold,
                                        text_threshold=text_threshold)

        annotated_frame = annotate(image_source=image_source,
                                boxes=boxes,
                                logits=logits,
                                phrases=phrases)

        # self._plot(image = annotated_frame)

        return boxes, logits, phrases
    

    def _export_annotations(self,
                            annotations:list,
                            labels_list:list,
                            names_list:list,
                            export_path:str)->None:
        
        '''
        Exports annotations from bounding boxes into txt files based on image names (yolo format)
        
        -Args:
            annotation (list): bounding box annotations
            labels_list (list): number encoded labels
            names_list (list): names of images (paths)
            export_path (str): folder path to export annotations
        -Returns:
            (None): exports annotation in txt
        '''

        os.makedirs(export_path, exist_ok=True)

        filenames = [os.path.splitext(os.path.basename(image_path))[0] for image_path in names_list]

        for annotation, labels, filename in zip(annotations, labels_list, filenames):
            output_filename = os.path.join(export_path, filename) + ".txt"

            with open(output_filename, "w") as file:
                for box, label in zip(annotation, labels):
                    # transform to list
                    list_box = box.tolist()
                    # add numerical label in the first position of the list
                    list_box.insert(0, label)
                    # round list values to 4 digits
                    list_box = [round(x,4) for x in list_box]
                    # convert each coordinate to str with one space between eachother
                    line = " ".join(str(num) for num in list_box)
                    # write line in file and add new line
                    file.write(line + "\n")
        

    def _map_labels(self, list_labels:list, last_num_label:int)->list:
        '''
        Maps string labels in number labels

        -Args:
            list_labels (list): labels of predicted objects
            last_num_label (int): 
        -Returns:
            mapped_list (list): transformed string labels to integers
        '''

        # Create a dictionary to store the mapping
        string_to_number = {}
        current_label = last_num_label

        # Iterate through all lists
        for sublist in list_labels:
            # Iterate through each item in the sublist
            for item in sublist:
                # If the string is not already in the dictionary, add it with the current label
                if item not in string_to_number:
                    string_to_number[item] = current_label
                    current_label += 1

        # Create a new list of lists with the mapped numbers
        mapped_list_of_lists = [
            [string_to_number[item] for item in sublist]
            for sublist in list_labels]

        return mapped_list_of_lists


    def execute_process(self):
        '''
        Executes a complete process for GroundingDINO predictions

        -Args:
            (None):
        -Returns:
            (None): 
        '''

        images_list = self._load_images(image_folder=self.IMAGES_FOLDER,
                                       extension=self.IMAGES_EXTENSION)
        
        for image in tqdm(images_list, desc="Progress", unit="item"):
            boxes, logits, phrases = self._infer(text_prompt=self.TEXT_PROMPT,
                                                image=image,
                                                box_threshold=self.BOX_THRESHOLD,
                                                text_threshold=self.TEXT_THRESHOLD)
            
            self.list_annotations.append(boxes)
            self.list_logits.append(logits)
            self.list_phrases.append(phrases)

        mapped_labels = self._map_labels(list_labels=self.list_phrases,
                                         last_num_label=self.LAST_NUM_LABEL)

        self._export_annotations(annotations=self.list_annotations,
                                labels_list=mapped_labels,
                                names_list=images_list,
                                export_path=self.EXPORT_PATH)