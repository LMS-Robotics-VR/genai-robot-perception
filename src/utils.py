import cv2
import numpy as np
import pyrealsense2 as rs
import os
import random
import shutil
from tqdm import tqdm
import yaml
from typing import Union



def image_capture(camera: Union[str,int],
                  export_path: str,
                  image_extension:str = "jpg")->None:
    '''
    Opens selected camera feed and saves frames in selected path when press "Enter"

    -Args:
        camera (str | int): "realsense" or int for usb camera
        export_path (str): path to export captured frames
        image_extension (str): extension of captured frames (jpg, png)
    -Returns:
        (None)
    '''

    count = 0

    if type(camera) == int:
        cap = cv2.VideoCapture(camera)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                cv2.imshow("Webcam feed", frame)

                key = cv2.waitKey(1)
                if key == 113:
                    break
                elif key == 13:
                    frame = cv2.resize(frame, (912,512))
                    cv2.imwrite(export_path+f"/frame{count}.{image_extension}", frame)
                    print("Captured frame")
                    count += 1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


    elif camera == "realsense":

        # realsense camera Class
        class DepthCamera:
            def __init__(self):
                # Configure depth and color streams
                self.pipeline = rs.pipeline()
                config = rs.config()

                # Get device product line for setting a supporting resolution
                pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
                pipeline_profile = config.resolve(pipeline_wrapper)
                device = pipeline_profile.get_device()
                device_product_line = str(device.get_info(rs.camera_info.product_line))

                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

                # Start streaming
                self.pipeline.start(config)

            def get_frame(self):
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                if not depth_frame or not color_frame:
                    return False, None, None
                return True, depth_image, color_image

            def release(self):
                self.pipeline.stop()

        # create instance of depth camera
        depth_camera = DepthCamera()

        while True:
            success, depth_image, frame = depth_camera.get_frame()

            if success:

                cv2.imshow("Intel RealSense D415 feed", frame)

                key = cv2.waitKey(1)
                if key == 113:
                    break
                elif key == 13:
                    frame = cv2.resize(frame, (912,512))
                    cv2.imwrite(export_path+f"/frame{count}.{image_extension}", frame)
                    print("Captured frame")
                    count += 1
            else:
                break

        depth_camera.release()
        cv2.destroyAllWindows()


def split_data(images_folder: str,
               labels_folder: str,
               output_folder: str,
               train_percent: float = 0.7,
               valid_percent: float = 0.2,
               test_percent: float = 0.1,
               image_extension: str = "jpg"):
    '''
    Splits images and labels in folder - subfolders:
        -train
        |____images
        |____labels
        -valid
        |____images
        |____labels
        -test
        |____images
        |____labels

    -Args:
        images_folder (str): path of image folder
        labels_folder (str): path of label folder
        output_folder (str): path to create train/valid/test folders
        train_percent (float): percent of train folder p ∈ (0,1)
        valid_percent (float): percent of valid folder p ∈ (0,1)
        test_percent (float): percent of test folder p ∈ (0,1)
        image_extension (str): extension of images
    -Returns:
        (None)
    '''

    # create train, valid, test folders
    for folder in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_folder, folder, 'images'))
        os.makedirs(os.path.join(output_folder, folder, 'labels'))

    # get list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(f'.{image_extension}')]

    # randomly shuffle the files
    random.shuffle(image_files)

    # calculate the number of files for each split
    total_files = len(image_files)
    train_count = int(total_files * train_percent)
    valid_count = int(total_files * valid_percent)
    test_count = total_files - train_count - valid_count

    # split the files
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]
    test_files = image_files[train_count + valid_count:]

    # copy images to corresponding folders
    for folder, files in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
        for file in files:
            # copy images
            shutil.copy(os.path.join(images_folder, file), os.path.join(output_folder,
                                                                        folder,
                                                                        'images',
                                                                        file))

            # copy labels
            label_file = file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder,
                                                                              folder,
                                                                              'labels',
                                                                              label_file))


def copy_files(source_dir:str,
               destination_dir:str)->None:
    '''
    Copies files from a source directory to a destination directory

    -Args:
        source_dir (str): path of source folder
        destination_dir (str): path of destination folder
    -Returns:
        (None)
    '''

    # ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # get a list of all files in the source directory
    files = os.listdir(source_dir)

    # copy each file to the destination directory
    for file in tqdm(files, desc="Copying files"):
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.copy2(source_path, destination_path)
        print(f"Copied: {file}")


def get_label_num(path_to_yaml:str)->int:
    '''
    Returns the number of classes from a yaml configuration file

    -Args:
        path_to_yaml (str): path to yaml configuration file
    -Returns:
        (int): number of classes
    '''

    with open(path_to_yaml, "r") as file:
        data = yaml.load(file, Loader=yaml.Loader)

        return data["nc"]
    

def arrange_in_sublists(url_list: list, sublist_size: int = 5)->list:
    '''
    Creates a list with sub-list elements, each of the sublists with
    max sublist_size elements

    -Args:
        url_list (list): a list which contains the urls of all images generations
        sublist_size (int): maximum number of elements of each sub-list
    -Returns:
        modified_list (list): a list with sub-lists of sublist_size elements
    '''
    modified_list = [url_list[i:i+sublist_size] for i in range(0, len(url_list), sublist_size)]
    return modified_list