import albumentations as A
import cv2
import glob
from PIL import Image
import os
import copy
import numpy as np
from typing import List, Tuple


class Augmentations:
    def __init__(self,
                 images_path,
                 annotations_path,
                 multiplier_rate,
                 export_images_path,
                 export_labels_path):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.multiplier_rate = multiplier_rate
        self.export_images_path = export_images_path
        self.export_labels_path = export_labels_path
        self.transform = A.Compose([
                ### SPATIAL TRANSOFORMATIONS ### 

                # A.Rotate(limit=20,
                #          mask_value=0,
                #          rotate_method="ellipse",
                #          border_mode=cv2.BORDER_CONSTANT,
                #          p = 0.4),
                A.Flip(p=1.0),
                A.PadIfNeeded(min_height=720,
                              min_width=1280,
                              p=1,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=0), # perform padding if image shape is lower that minimum
                #A.RandomRotate90(p=1.0), # randomly rotate the input by 90 degrees zero or more times.
                #A.Resize(height=853, width=1280, p=1.0), # resize in specific height and width
                A.VerticalFlip(p=0.4), # flip the input vertically around the x-axis. 

                ### PIXEL TRANSFORMATIONS ###

                A.ChannelShuffle(p=0.3),
                #A.CLAHE(clip_limit = (3,4), p = 1.0), # histogram equalization
                A.RandomBrightnessContrast(brightness_limit=0.10,
                                           contrast_limit=0.10,
                                           p=0.3), # random brightess and contrast using limit values
                A.ToGray(p=0.25),

            ], bbox_params=A.BboxParams(format='yolo'))


    def _load_images(self,
                     path_of_images: str)->List[np.ndarray]:
        
        '''
        Function which loads the images from a folder path

        -Args: 
            path_of_images (str): a folder path which contains the images
        -Returns:
            (list): list with images of np.ndarray format 
        '''
        return [cv2.imread(file) for file in sorted(glob.glob(path_of_images + "/*.jpg"))]


    def _load_annotations(self,
                          path_of_annotations: str
                          )->List[List[Tuple[float,float,float,float,int]]]:

        '''
        Function which loads annotations from a folder path in format required by augmentations
        Augmentations format -> x y w h class
        
        -Args:
            path_of_annotations (str): a folder path which contains the annotations
        -Returns:
            all_annotations (list): list of annotations 
        '''

        # create empty list to load annotations
        all_annotations = []

        for txt in sorted(glob.glob(path_of_annotations + "/*.txt")):
            annotations = [] # clean this list in each iteration
            with open(txt, "r") as file:
                for line in file:
                    line_values = line.strip().split(" ")
                    class_label = int(line_values[0])
                    x1,y1,x2,y2 = map(float, line_values[1:])
                     # transform annotations in such format needed by the albumentations library
                    annotations.append((x1,y1,x2,y2,class_label))
                all_annotations.append(annotations)

        return all_annotations


    def _transform_images_and_annotations(self,
                                          transform_func: A.Compose,
                                          images: list,
                                          annotations: list,
                                          multiplier_rate: str = 1
                                          )->Tuple[List[np.ndarray], list]:
        
        '''
        -Args:
            transform_func: a defined function that performs the transformations
            images: a list of numpy.array images
            annotations: a list with annotations for each image
        -Returns:
            transformed_images (list): list with transformed images
            transformed_bboxes (list): list with transformed annotations
        '''

        transformed_images = []
        transformed_bboxes = []

        for _ in range(multiplier_rate):
            for image, annotation in zip(images, annotations):
                transformed = transform_func(image = image, bboxes = annotation)
                transformed_images.append(transformed["image"])
                transformed_bboxes.append(transformed["bboxes"])

        return transformed_images, transformed_bboxes


    def _plot_tranformed_img_bboxes(self,
                                    images: List[np.ndarray],
                                    annotations: list,
                                    img_num: int)->None:
        
        '''
        Plots an images from a list of images
        
        -Args:
            images (list): a list with images in np.ndarray format
            annotations (list): a list with annotations
            img_num (int): image number to plot
        -Returns:
            (None)
        '''

        transformed_images = copy.deepcopy(images)
        transformed_bboxes = copy.deepcopy(annotations)

        # convert the annotations from YOLO format to image coordinates
        image_height, image_width, _ = transformed_images[img_num].shape
        scaled_annotations = []
        for annotation in transformed_bboxes[img_num]:
            x_center, y_center, width, height, class_label = annotation

            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            scaled_annotations.append((x_min, y_min, x_max, y_max, class_label))

        # plot the image with bounding boxes
        for annotation in scaled_annotations:
            x_min, y_min, x_max, y_max, class_label = annotation

            # draw the bounding box on the image
            cv2.rectangle(transformed_images[img_num],
                          (x_min, y_min),
                          (x_max, y_max),
                          (0, 255, 0),
                          2)
            # draw class label
            cv2.putText(transformed_images[img_num],
                        str(class_label),
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)
           

        # display the image with bounding boxes
        cv2.imshow('Transformed image', transformed_images[img_num])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _save_transformed_images(self,
                                save_images_path: str,
                                list_of_images: List[np.ndarray])->None:

        '''
        Function which saves all images in a specific path folder

        -Args:
            save_images_path (str): folder path where you want the images to get exported
            list_of_images (): list in which images are stored as np.ndarray
        -Returns:
            (None)
        '''

        # create the output directory if it doesn't exist
        os.makedirs(save_images_path, exist_ok = True)

        img_num = 0
        for image in list_of_images:
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(save_images_path + f"/img_{img_num}.jpg")
            img_num += 1


    def _save_transformed_annotations(self,
                                      save_annotation_path: str,
                                      list_of_transformed_annotations: list)->None:

        '''
        Saves transformed annotations in a specific folder path in txt
        (yolo format: class, x, y, w, h)

        -Args:
            save_annotations_path (str): folder path in which annotation will be exported
            list_of_transformed_annotations (list): list which contains transformed annotations 
                            (each element in the list is also a list which contains 
                            tuples with annotations coordinates)
        -Returns:
            (None)
        '''

        # create the output directory if it doesn't exist
        os.makedirs(save_annotation_path, exist_ok = True)

        txt_num = 0
        for annotation in list_of_transformed_annotations:

            # rearrange the position of each annotation (before: x_center, y_center, width, height, class_label ---> after: class_label, x_center, y_center, width, height)
            bboxes_rearranged = [(inner_tuple[-1], ) + inner_tuple[:-1] for inner_tuple in annotation]

            # construct the full file path
            output_file = os.path.join(save_annotation_path, f'img_{txt_num}.txt')

            # Open the file in write mode
            with open(output_file, 'w') as file:

                for tuple_values in bboxes_rearranged:
                    # Convert the tuple elements to strings and join them with a space
                    line = ' '.join(str(num) for num in tuple_values)
                    # Write the line to the file
                    file.write(line + "\n")

            txt_num += 1


    def augment(self)->None:
        '''
        Runs methods for augmentation process

        -Args:
            (None)
        -Returns:
            (None)
        '''
        
        list_images = self._load_images(path_of_images=self.images_path)
        list_annotations = self._load_annotations(path_of_annotations=self.annotations_path)
        transformed_images, transformed_bboxes = self._transform_images_and_annotations(
                                                    transform_func=self.transform,
                                                    images=list_images,
                                                    annotations=list_annotations,
                                                    multiplier_rate=self.multiplier_rate)
        # self._plot_tranformed_img_bboxes(images=transformed_images,
        #                                 annotations=transformed_bboxes,
        #                                 img_num=0)
        self._save_transformed_images(save_images_path=self.export_images_path,
                                     list_of_images=transformed_images)
        self._save_transformed_annotations(save_annotation_path=self.export_labels_path,
                                          list_of_transformed_annotations=transformed_bboxes)