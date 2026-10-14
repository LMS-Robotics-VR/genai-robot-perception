import os
import glob
import argparse
import yaml
from tqdm import tqdm
from time import sleep
from leonardo_api import generate_images, set_api_key
from GDINO_inference import GroundingDino
from augmentations import Augmentations
from yolo_train import Trainer
from modify_yaml import YamlModifier
from utils import image_capture, split_data, copy_files, get_label_num, arrange_in_sublists


# root of the repository, used to resolve every relative path of the
# configuration file, so the pipeline can be launched from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# folder which holds every model (GroundingDINO config and weights, YOLO weights)
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "src", "models")

# default configuration file, sitting in the root of the repository
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")


def parse_args():
    '''
    Parses the command line arguments of the pipeline

    -Args:
        (None)
    -Returns:
        (argparse.Namespace): parsed arguments
    '''

    parser = argparse.ArgumentParser(description="Dataset generation and YOLO training pipeline")
    parser.add_argument("--config",
                        default=DEFAULT_CONFIG,
                        help="path to the yaml configuration file "
                             "(default: config.yaml in the root of the repository)")

    return parser.parse_args()


def resolve_path(path:str)->str:
    '''
    Turns a path of the configuration file into an absolute path. Relative
    paths are resolved against the root of the repository, absolute paths are
    left untouched

    -Args:
        path (str): path as written in the configuration file
    -Returns:
        (str): absolute path
    '''

    if not path:
        return path

    return os.path.normpath(os.path.join(PROJECT_ROOT, os.path.expanduser(path)))


def resolve_model_path(path:str)->str:
    '''
    Turns a model path of the configuration file into an absolute path. A bare
    filename is looked up inside the models folder, a path with a separator is
    resolved like any other path of the configuration file

    -Args:
        path (str): model filename or path as written in the configuration file
    -Returns:
        (str): absolute path of the model
    '''

    if not path:
        return path

    if os.path.dirname(path) == "":
        return os.path.join(MODELS_FOLDER, path)

    return resolve_path(path)


def load_config(config_path:str)->dict:
    '''
    Loads the yaml configuration file

    -Args:
        config_path (str): path to the yaml configuration file
    -Returns:
        (dict): configuration parameters
    '''

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.load(file, Loader=yaml.Loader)


def first_set(*values):
    '''
    Returns the first value which is not empty (None or "").
    Used to let a configuration field fall back to another one.

    -Args:
        *values: candidate values, in priority order
    -Returns:
        the first value which is not None and not an empty string
    '''

    for value in values:
        if value is not None and value != "":
            return value

    return None


def ask_if_empty(value, question:str)->str:
    '''
    Returns the configured value, or asks the user for it when it is empty

    -Args:
        value: value coming from the configuration file
        question (str): question shown to the user when the value is empty
    -Returns:
        (str): the configured or the typed value
    '''

    if value:
        return value

    return input(question)


def resolve_config(config:dict)->dict:
    '''
    Fills in every parameter which is left empty in the configuration file with
    the default described in config.yaml, so the folders of the pipeline stay
    chained (capture -> generate -> annotate -> augment)

    -Args:
        config (dict): configuration parameters as loaded from the yaml file
    -Returns:
        config (dict): configuration parameters with the empty fields resolved
    '''

    capture = config["capture"]
    leonardo = config["leonardo"]
    gdino = config["grounding_dino"]
    augment = config["augmentation"]
    dataset = config["dataset"]

    # the extension of the captured frames is the default for every later step
    extension = capture["image_extension"]

    # Leonardo reads the captured frames and writes next to them
    leonardo["api_key"] = first_set(leonardo["api_key"],
                                    os.environ.get("LEONARDO_API_KEY"))
    leonardo["reference_images_path"] = first_set(leonardo["reference_images_path"],
                                                  capture["export_path"])
    leonardo["reference_images_extension"] = first_set(leonardo["reference_images_extension"],
                                                       extension)
    leonardo["export_path"] = first_set(leonardo["export_path"],
                                        leonardo["reference_images_path"])

    # GroundingDINO annotates whatever Leonardo produced
    gdino["images_path"] = first_set(gdino["images_path"], leonardo["export_path"])
    gdino["images_extension"] = first_set(gdino["images_extension"], extension)

    # the augmentation reads and overwrites the annotated folders
    augment["images_path"] = first_set(augment["images_path"], gdino["images_path"])
    augment["annotations_path"] = first_set(augment["annotations_path"],
                                            gdino["annotations_export_path"])
    augment["export_images_path"] = first_set(augment["export_images_path"],
                                              augment["images_path"])
    augment["export_labels_path"] = first_set(augment["export_labels_path"],
                                              augment["annotations_path"])

    # the train/valid/test folders are created next to the collected images
    dataset["images_extension"] = first_set(dataset["images_extension"], extension)
    dataset["output_folder"] = first_set(dataset["output_folder"],
                                         os.path.dirname(dataset["images_path"]))

    # a bare filename is resolved inside the dataset output folder
    yaml_file = dataset["yaml_file"]
    if os.path.dirname(yaml_file) == "":
        yaml_file = os.path.join(dataset["output_folder"], yaml_file)
    dataset["yaml_file"] = yaml_file

    # every path becomes absolute, so the pipeline can run from any directory
    for section, fields in (("capture", ["export_path"]),
                            ("leonardo", ["reference_images_path", "export_path"]),
                            ("grounding_dino", ["images_path", "annotations_export_path"]),
                            ("augmentation", ["images_path", "annotations_path",
                                              "export_images_path", "export_labels_path"]),
                            ("dataset", ["images_path", "labels_path",
                                         "output_folder", "yaml_file"])):
        for field in fields:
            config[section][field] = resolve_path(config[section][field])

    # models are looked up inside the models folder
    gdino["config_path"] = resolve_model_path(gdino["config_path"])
    gdino["weights_path"] = resolve_model_path(gdino["weights_path"])
    config["training"]["model_weights"] = resolve_model_path(config["training"]["model_weights"])

    # prompts are asked interactively when they are not in the configuration
    leonardo["text_prompt"] = ask_if_empty(leonardo["text_prompt"],
                                           "Please provide TEXT PROMPT for Leonardo: ")
    gdino["text_prompt"] = ask_if_empty(
        gdino["text_prompt"],
        "Please provide PREDICT TEXT PROMPT for GroundingDINO detection: ")

    return config


def main():

    args = parse_args()
    config = resolve_config(load_config(args.config))

    capture = config["capture"]
    leonardo = config["leonardo"]
    gdino = config["grounding_dino"]
    augment = config["augmentation"]
    dataset = config["dataset"]
    training = config["training"]
    pipeline = config["pipeline"]

    delay = pipeline["step_delay"]
    data_yaml = dataset["yaml_file"]

    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}. "
                                "Create it before running the pipeline "
                                "(see the Configuration section of the README).")

    grounding_dino = GroundingDino(config_path=gdino["config_path"],
                                   weights_path=gdino["weights_path"],
                                   images_folder=gdino["images_path"],
                                   images_extension=gdino["images_extension"],
                                   box_threshold=gdino["box_threshold"],
                                   text_threshold=gdino["text_threshold"],
                                   text_prompt=gdino["text_prompt"],
                                   export_path=gdino["annotations_export_path"],
                                   last_num_label=get_label_num(data_yaml))

    augmentor = Augmentations(images_path=augment["images_path"],
                              annotations_path=augment["annotations_path"],
                              multiplier_rate=augment["multiplier_rate"],
                              export_images_path=augment["export_images_path"],
                              export_labels_path=augment["export_labels_path"])

    modifier = YamlModifier(file_path=data_yaml)

    trainer = Trainer(model_weights=training["model_weights"])

    # the capture and the generation steps write with cv2 / requests, which do
    # not create their destination folder on their own
    os.makedirs(capture["export_path"], exist_ok=True)
    os.makedirs(leonardo["export_path"], exist_ok=True)

    if capture["enabled"]:
        image_capture(camera=capture["camera"],
                      export_path=capture["export_path"],
                      image_extension=capture["image_extension"])

        print("Images captured!")
        sleep(delay)

    if leonardo["enabled"]:
        print("Sending captured images to Leonardo API")
        sleep(delay)

        set_api_key(leonardo["api_key"])

        # name distinguisher for image export
        distinguisher = 0

        reference_images = sorted(glob.glob(leonardo["reference_images_path"] +
                                            f"/*.{leonardo['reference_images_extension']}"))
        batch_size = leonardo["images_per_batch"]
        sublisted_ref_images = arrange_in_sublists(reference_images, batch_size)

        for ref_list in tqdm(sublisted_ref_images, desc="Progress", unit="item"):
            generate_images(ref_image_list=ref_list,
                            distinguisher=distinguisher,
                            text_prompt=leonardo["text_prompt"],
                            destination_path=leonardo["export_path"],
                            generation=leonardo["generation"],
                            wait_seconds=leonardo["wait_seconds"])
            distinguisher += batch_size

        print("Generating images completed!")
        sleep(delay)

    print("Images will iterate through GroundingDINO")
    sleep(delay)

    grounding_dino.execute_process()

    print("Predictions from GroundingDINO completed!")
    sleep(delay)

    if pipeline["review_annotations"]:
        print("Please make sure predictions are correct! Modify them if needed.")
        print(f"Image folder path: {gdino['images_path']}")
        print(f"Label folder path: {gdino['annotations_export_path']}")

        while input("Type 'finished' to continue with next process:") != "finished":
            pass

    if augment["enabled"]:
        print("Augmenting images and labels")
        sleep(delay)

        augmentor.augment()

        print("Augmentation finished!")
        sleep(delay)

    print("Copying all images and labels to main folders")
    sleep(delay)

    copy_files(source_dir=augment["export_images_path"],
               destination_dir=dataset["images_path"])
    print("Copying images finished!")

    copy_files(source_dir=augment["export_labels_path"],
               destination_dir=dataset["labels_path"])
    print("Copying labels finished!")
    sleep(delay)

    print("Splitting images and labels in train/valid/test folders")
    sleep(delay)

    split_data(images_folder=dataset["images_path"],
               labels_folder=dataset["labels_path"],
               output_folder=dataset["output_folder"],
               train_percent=dataset["split"]["train"],
               valid_percent=dataset["split"]["valid"],
               test_percent=dataset["split"]["test"],
               image_extension=dataset["images_extension"])

    print("Image and label split completed!")
    sleep(delay)
    print("Creating yaml file for training")
    sleep(delay)

    modifier.modify_yaml(new_label=gdino["text_prompt"])

    print("yaml file ready!")
    sleep(delay)

    if training["enabled"]:
        print("Process continues with model training. Drink a coffee...")
        sleep(delay)

        trainer.train(data=data_yaml,
                      epochs=training["epochs"],
                      imgsz=training["imgsz"],
                      batch=training["batch"],
                      save=training["save"],
                      device=training["device"],
                      lr0=training["lr0"],
                      lrf=training["lrf"],
                      half=training["half"])

        print("Training completed!")
        sleep(delay)

    print("Process completed!!")
    sleep(delay)


if __name__ == "__main__":
    main()
