import os
import random
import shutil

def split_data(images_folder,
               labels_folder,
               output_folder,
               train_percent,
               valid_percent,
               test_percent,
               image_extension):
    # Create train, valid, test folders
    for folder in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_folder, folder, 'images'))
        os.makedirs(os.path.join(output_folder, folder, 'labels'))

    # Get list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(f'.{image_extension}')]

    # Randomly shuffle the files
    random.shuffle(image_files)

    # Calculate the number of files for each split
    total_files = len(image_files)
    train_count = int(total_files * train_percent)
    valid_count = int(total_files * valid_percent)
    test_count = total_files - train_count - valid_count

    # Split the files
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]
    test_files = image_files[train_count + valid_count:]

    # Copy images to corresponding folders
    for folder, files in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
        for file in files:
            # Copy images
            shutil.copy(os.path.join(images_folder, file), os.path.join(output_folder,
                                                                        folder,
                                                                        'images',
                                                                        file))

            # Copy labels
            label_file = file.replace('.jpg', '.txt')
            shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_folder,
                                                                              folder,
                                                                              'labels',
                                                                              label_file))