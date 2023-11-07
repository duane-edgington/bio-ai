import time
import tator
from dotenv import load_dotenv
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

from bio.db.tator_db import init_api, find_project
from bio.logger import create_logger_file, info, err

# Default values
# The base directory is the same directory as this file
DEFAULT_BASE_DIR = Path(__file__).parent.as_posix()

DEFAULT_VERSION = 'Baseline'
DEFAULT_PROJECT = '901103-biodiversity'

voc_xml_path = Path('/Users/dcline/Dropbox/code/bio-ai/CtenophoraData')
image_url_base = 'http://digits-dev-box-fish.shore.mbari.org:8080/Ctenophora_sp_A_confused/'

def is_valid_url(url):
    try:
        response = requests.get(url)
        return response.status_code == 200  # HTTP status code 200 indicates success
    except requests.RequestException:
        return False  # URL is not reachable or invalid

#  Convert the xml data to a localization object formatted for the database
def voc_to_obj(xml_file, box_type='box', media_id=0, project_id=0):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the image width and height
    image_width = 0
    image_height = 0
    for size in root.findall('size'):
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        obj_info = {}
        obj_info['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_info['xmin'] = float(bbox.find('xmin').text)
        obj_info['ymin'] = float(bbox.find('ymin').text)
        obj_info['xmax'] = float(bbox.find('xmax').text)
        obj_info['ymax'] = float(bbox.find('ymax').text)

        # Normalize the bounding box to the image size
        obj_info['xmin'] = obj_info['xmin'] / image_width
        obj_info['ymin'] = obj_info['ymin'] / image_height
        obj_info['xmax'] = obj_info['xmax'] / image_width
        obj_info['ymax'] = obj_info['ymax'] / image_height
        obj_info['obs_uuid'] = Path(xml_file).stem
        obj_info['concept'] = obj_info['name']
        obj_info['score'] = 1
        obj_info['Label'] = obj_info['name']
        obj_info['generator'] = 'vars-annotation'

        x = obj_info['xmin']
        y = obj_info['ymin']
        w = obj_info['xmax'] - obj_info['xmin']
        h = obj_info['ymax'] - obj_info['ymin']

        # if the height or width is < 0 skip this object
        if w < 0 or h < 0:
            continue

        attributes = {
            'Label': obj_info['name'],
            'score': 1.0,
            'concept': obj_info['name'],
            'obs_uuid': Path(xml_file).stem,
            'cluster': -1,
            'group': 'CTENOPHORA_SP_A_CONFUSED',
            'generator': 'vars-annotation'
        }

        out = {
            'type': box_type,
            'media_id': media_id,
            'project': project_id,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'frame': 0,
            'attributes': attributes
        }
        objects.append(out)

    return objects


if __name__ == '__main__':
    create_logger_file(Path.cwd(), 'load_voc')
    load_dotenv('.env')
    start_time = time.time()
    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    # Find the media type
    media_types = api.get_media_type_list(project.id)
    image_type = None
    media_to_load = []

    for m in media_types:
        if m.dtype == "image":
            image_type = m.id
            break

    # Get the localization type
    localization_types = api.get_localization_type_list(project.id)

    # the box type is the one with the name 'Boxes'
    box_type = None
    for l in localization_types:
        if l.name == 'Boxes':
            box_type = l.id
            break

    # Grab all the localizations in the project
    for xml_file in voc_xml_path.rglob('*.xml'):

        # form the image url
        for ext in ['.jpg', '.png']:
            image_name = xml_file.name.replace('.xml', ext)
            image_url = image_url_base + image_name
            if is_valid_url(image_url):
                break

        # upload the image url reference
        info(f'Importing image url {image_url}')
        error = False
        try:
            for progress, response in tator.util.import_media(api,
                                               image_type,
                                               image_url,
                                               attributes={},
                                               fname=image_name):
                info(f"Creating progress: {progress}%")
                info(f'Uploading {image_url}')
        except Exception as e:
            err(f'Error: {e}')
            if 'already exists' in str(e):
                info(f'Image {image_url} already exists')
            if 'not found' in str(e).lower():
                err(f'Image {image_url} not found')
                error = True

        if error:
            continue

        # Get the image id
        media = api.get_media_list(project=project.id, name=image_name)
        if len(media) == 0:
            err(f'Could not find image {image_name}')
            continue

        image_id = media[0].id

        # load the localizations
        info(f'Loading localizations from {xml_file}')
        boxes = voc_to_obj(xml_file, box_type=box_type, media_id=image_id, project_id=project.id)
        response = api.create_localization_list(project.id, boxes)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done. Elapsed time: {elapsed_time}')
