import json
import requests
import time
import os

# API key, read from the environment by default and overridden by the
# leonardo.api_key field of the configuration file through set_api_key()
api_key = os.environ.get("LEONARDO_API_KEY", "")

# default payload of the /generations endpoint, overridden by the
# leonardo.generation section of the configuration file
DEFAULT_GENERATION = {
    "model_id": "1e60896f-3c26-4296-8ecc-53e2afecc132",
    "width": 912,
    "height": 512,
    "num_images": 4,
    "alchemy": False,
    "preset_style": "CINEMATIC",
    "init_strength": 0.6
}


def set_api_key(key:str)->None:
    '''
    Sets the API key used for every call to Leonardo

    -Args:
        key (str): the Leonardo API key
    -Returns:
        (None)
    '''

    global api_key

    if not key:
        raise ValueError("No Leonardo API key provided. Set it in config.yaml "
                         "(leonardo.api_key) or in the LEONARDO_API_KEY "
                         "environment variable.")

    api_key = key


def download_image(url:str, destination:str)->None:
    '''
    Downloads images from provided url and exports in specific folder

    -Args:
        url (str): url of image
        destination (str): path to export
    -Returns:
        (None) Exports image to destination path
    '''

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print("Image downloaded successfully!")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


def get_filename_from_url(url:str, distinguisher:int)->str:
    '''
    Get the filename from a given url.
    e.g. https://google.com/images/path/to/image/generated_img.jpg --> /generated_img.jpg
    A distinguisher number is used to avoid same name of images and result to override of image

    -Args:
        url (str): url of image
        distinguisher (int): an integer which will be added before extension to avoid duplicate
                            exports
    -Returns:
        (str): filename of an image
    '''

    # Split the URL by '/' to get the parts
    url_parts = url.split('/')
    
    # The last part of the URL is the filename with extension
    filename_with_extension = url_parts[-1]
    
    # Use os.path.splitext to separate the filename and extension
    filename, extension = os.path.splitext(filename_with_extension)
    
    return os.path.join("/", filename + str(distinguisher) + extension)


def generate_images(ref_image_list:list,
                    distinguisher:int,
                    text_prompt:str,
                    destination_path:str,
                    generation:dict = None,
                    wait_seconds:int = 45)->None:
    '''
    Generates images from calling Leonardo API
    -Args:
        ref_image_list (list): an image path which will be passed as reference image to API
        distinguisher (str): an integer value which is used in image export name to avoid
                            duplicate image names when downloading them from API
        text_prompts (str): text prompt to be passed in payload for Leonardo API
        destination_path (str): destination path to download images from API
        generation (dict): generation settings (model_id, width, height, num_images,
                            alchemy, preset_style, init_strength). Defaults to
                            DEFAULT_GENERATION for the keys which are not given
        wait_seconds (int): seconds to wait for the generation jobs to finish
    -Returns:
        (None) : downloads generated images from API
    '''
    settings = dict(DEFAULT_GENERATION)
    if generation:
        settings.update(generation)

    generated_urls = []
    print(ref_image_list)
    for ref_image in ref_image_list:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer %s" % api_key
        }

        # Get a presigned URL for uploading an image
        url = "https://cloud.leonardo.ai/api/rest/v1/init-image"

        payload = {"extension": "jpg"}

        response = requests.post(url, json=payload, headers=headers)

        print(response.status_code)

        # Upload image via presigned URL
        fields = json.loads(response.json()['uploadInitImage']['fields'])

        url = response.json()['uploadInitImage']['url']

        image_id = response.json()['uploadInitImage']['id']  # For getting the image later

        image_file_path = ref_image
        files = {'file': open(image_file_path, 'rb')}

        response = requests.post(url, data=fields, files=files) # Header is not needed

        print(response.status_code)

        # Generate with an image prompt
        url = "https://cloud.leonardo.ai/api/rest/v1/generations"

        payload = {
            "height": settings["height"],
            "modelId": settings["model_id"],
            "prompt": text_prompt,
            "width": settings["width"],
            "alchemy": settings["alchemy"],
            "num_images": settings["num_images"],
            "init_image_id": image_id,
            "presetStyle": settings["preset_style"],
            "init_strength": settings["init_strength"]
        }

        response = requests.post(url, json=payload, headers=headers)

        print(response.status_code)
        print(response.json())
        # Get the generation of images
        generation_id = response.json()['sdGenerationJob']['generationId']


        generated_urls.append("https://cloud.leonardo.ai/api/rest/v1/generations/%s" % generation_id)

    time.sleep(wait_seconds)

    for i, url in enumerate(generated_urls):
        response = requests.get(url, headers=headers)
        response.json()
        print(response.text)


    ###############################################

        for image in response.json()["generations_by_pk"]["generated_images"]:
            print(image["url"])


        response.json()["generations_by_pk"]["generated_images"]
        for link in response.json()["generations_by_pk"]["generated_images"]:
            download_image(url=link["url"],
                        destination=destination_path+get_filename_from_url(url=link["url"],
                                                                            distinguisher=distinguisher+i))
            
        print(f"Generation for ref image {distinguisher+i} ---> Completed")