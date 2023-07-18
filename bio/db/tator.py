import json
import multiprocessing
import os
import pickle
import tempfile
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import tator
import io
import itertools
import torch
from torchvision.ops import nms
from pascal_voc_writer import Writer

from bio.model import KClassify
from bio.data.generator import create_cifar_dataset
from bio.logger import info, debug, err, exception, warn


def init_api() -> tator.api:
    """
    Initialize the tator api
    :return: tator.api
    """
    if 'TATOR_API_HOST' not in os.environ:
        err('TATOR_API_HOST not found in environment variables!')
        return
    if 'TATOR_API_TOKEN' not in os.environ:
        err('TATOR_API_TOKEN not found in environment variables!')
        return

    try:
        api = tator.get_api(os.environ['TATOR_API_HOST'], os.environ['TATOR_API_TOKEN'])
        info(api)
        return api
    except Exception as e:
        exception(e)
        exit(-1)


def find_project(api: tator.api, project_name: str) -> tator.models.Project:
    """
    Find a project by name
    :param api: tator.api
    :param project_name: Name of the project
    :return: tator.models.Project
    """
    try:
        projects = api.get_project_list()
        info(projects)

        # Find the project by name
        project = [p for p in projects if p.name == project_name]
        if len(project) == 0:
            err(f'Could not find project {project_name}')
            return

        p = project[0]
        return p
    except Exception as e:
        exception(e)
        exit(-1)
    return None


def get_media(api: tator.api, project_id: int, media_ids: []):
    """
    Get media from a project
    :param api: tator.api
    :param project_id: project id
    :param media_ids: List of media ids
    """
    try:
        medias = []
        for start in range(0, len(media_ids), 200):
            new_medias = api.get_media_list(project=project_id, media_id=media_ids[start:start + 200])
            medias = medias + new_medias
        return medias
    except Exception as e:
        err(e)
        return None


def download_data(api: tator.api,
                  project_id: int,
                  group: str,
                  version: str,
                  generator: str,
                  output_path: Path,
                  concept_list: [],
                  cifar_size: int = 32,
                  voc: bool = False,
                  coco: bool = False,
                  cifar: bool = False):
    """
    Download a dataset based on a version tag for training
    :param api: tator.api
    :param project_id: project id
    :param group: group name
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param output_path: output directory to save the dataset
    :param concept_list: (optional) list of concepts to download
    :param cifar_size: (optional) size of the CIFAR images
    :param voc: (optional) True if the dataset should also be stored in VOC format
    :param coco: (optional) True if the dataset should also be stored in COCO format
    :param cifar: (optional) True if the dataset should also be stored in CIFAR format
    """
    try:
        # Get the version
        versions = api.get_version_list(project=project_id)
        debug(versions)

        # Find the version by name
        version_match = [v for v in versions if v.name == version]
        if len(version_match) == 0:
            err(f'Could not find version {version}')
            return

        version = version_match[0]
        info(version)

        # TODO: figure out out to get annotations for a specific version
        attribute_filter = []
        num_records = 0
        if generator:
            attribute_filter = [f"generator::{generator}"]
        if group:
            attribute_filter += [f"group::{group}"]
        if concept_list:
            for c in concept_list:
                num_records += api.get_localization_count(project=project_id, attribute=attribute_filter + [f"concept::{c}"])
        else:
            num_records = api.get_localization_count(project=project_id, attribute=attribute_filter)

        info(f'Found {num_records} records for version {version.name} and generator {generator}, group {group} and '
             f"including {concept_list if concept_list else 'everything'} ")

        if num_records == 0:
            info(f'Could not find any records for version {version.name} and generator {generator}, group {group} and '
                 f"including {concept_list if concept_list else 'everything'} ")
            return

        # Create the output directory in the expected format that deepsea-ai expects for training
        # See https://docs.mbari.org/deepsea-ai/data/ for more information
        label_path = output_path / 'labels'
        label_path.mkdir(exist_ok=True)
        media_path = output_path / 'images'
        media_path.mkdir(exist_ok=True)
        if voc:
            voc_path = output_path / 'voc'
            voc_path.mkdir(exist_ok=True)
        if coco:
            coco_path = output_path / 'coco'
            coco_path.mkdir(exist_ok=True)

        localizations = []
        inc = min(500, num_records)
        if concept_list:
            for c in concept_list:
                for start in range(0, num_records, inc):
                    filter = attribute_filter + [f"concept::{c}"]
                    info(f'Query records {start} to {start + inc} using attribute filter {filter} ')

                    new_localizations = api.get_localization_list(project=project_id,
                                                                  attribute=filter,
                                                                  start=start,
                                                                  stop=start + 500)

                    if len(new_localizations) == 0:
                        break

                    localizations = localizations + new_localizations
        else:
            for start in range(0, num_records, inc):
                info(f'Query records {start} to {start + inc} using attribute filter {attribute_filter}')

                new_localizations = api.get_localization_list(project=project_id,
                                                              attribute=attribute_filter,
                                                              start=start,
                                                              stop=start + inc)

                if len(new_localizations) == 0:
                    break

                localizations = localizations + new_localizations

        info(f'Found {len(localizations)} records for version {version.name} and generator {generator}')
        info(f'Creating output directory {output_path} in YOLO format')

        media_lookup_by_id = {}

        # Get all the unique media ids in the localizations
        media_ids = list(set([l.media for l in localizations]))

        # Get all the media objects at those ids
        all_media = get_media(api, project_id, media_ids)

        # Get all the unique media names
        media_names = list(set([m.name.split('.png')[0] for m in all_media]))

        # Get all the unique Label attributes and sort them alphabetically
        labels = list(sorted(set([l.attributes['Label'] for l in localizations])))

        # Write the labels to a file called labels.txt
        with (output_path / 'labels.txt').open('w') as f:
            for label in labels:
                f.write(f'{label}\n')

        # Download all the media files - this needs to be done before we can create the VOC files which reference the
        # media file size
        for m in media_ids:
            media = api.get_media(m)
            out_path = media_path / media.name
            if not out_path.exists():
                for progress in tator.util.download_media(api, media, out_path):
                    debug(f"{media.name} download progress: {progress}%")

        # Create YOLO, and optionally COCO, CIFAR, or VOC formatted files

        info(f'Creating YOLO files in {label_path}')
        json_content = {}
        if voc:
            info(f'Creating VOC files in {voc_path}')

        for media_name in media_names:

            # Get all the localizations for this media
            media_localizations = [l for l in localizations if l.media == media.id]

            # Get the media object
            media = [m for m in all_media if m.name.split('.png')[0] == media_name][0]

            media_lookup_by_id[media.id] = media_path / media.name
            yolo_path = label_path / f'{media_name}.txt'

            with yolo_path.open('w') as f:

                for loc in media_localizations:
                    # Get the label index
                    label_idx = labels.index(loc.attributes['Label'])

                    # Get the bounding box which is normalized to a 0-1 range and centered
                    # f.write(f'{label_idx} {loc.x} {loc.y} {loc.width} {loc.height}\n')
                    x = loc.x + loc.width / 2
                    y = loc.y + loc.height / 2
                    w = loc.width
                    h = loc.height
                    f.write(f'{label_idx} {x} {y} {w} {h}\n')

            # optionally create VOC files
            if voc:
                # Paths to the VOC file and the image
                voc_xml_path = voc_path / f'{media_name}.xml'
                image_path = (media_path / media.name).as_posix()

                # Get the image size from the image using PIL
                image = Image.open(image_path)
                image_width, image_height = image.size

                writer = Writer(image_path, image_width, image_height)

                # Add localizations
                for loc in media_localizations:

                    # Get the bounding box which is normalized to the image size and upper left corner
                    x1 = loc.x
                    y1 = loc.y
                    x2 = loc.x + loc.width
                    y2 = loc.y + loc.height

                    x1 *= image_width
                    y1 *= image_height
                    x2 *= image_width
                    y2 *= image_height

                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))

                    writer.addObject(loc.attributes['Label'], x1, y1, x2, y2)

                # Write the file
                writer.save(voc_xml_path.as_posix())

            if coco:
                coco_localizations = []
                # Add localizations
                for loc in media_localizations:
                    # Get the bounding box which is normalized to the image size and upper left corner
                    x = loc.x
                    y = loc.y
                    w = loc.x + loc.width
                    h = loc.y + loc.height

                    x *= image_width
                    y *= image_height
                    w *= image_width
                    h *= image_height

                    x = int(round(x))
                    y = int(round(y))
                    w = int(round(w))
                    h = int(round(h))

                    # optionally add to COCO formatted dataset
                    coco_localizations.append({
                        'concept': loc.attributes['Label'],
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                    })

                json_content[yolo_path.as_posix()] = coco_localizations

        # optionally create a CIFAR formatted dataset
        if cifar:
            cifar_path = output_path / 'cifar'
            cifar_path.mkdir(exist_ok=True)
            info(f'Creating CIFAR data in {cifar_path}')

            images, labels = create_cifar_dataset(cifar_size, cifar_path, media_lookup_by_id, localizations, labels)
            np.save(cifar_path / 'images.npy', images)
            np.save(cifar_path / 'labels.npy', labels)

        if coco:
            info(f'Creating COCO data in {coco_path}')
            with (coco_path / 'coco.json').open('w') as f:
                json.dump(json_content, f, indent=2)

    except Exception as e:
        exception(e)
        exit(-1)


def predict_classify_top1(temp_path: Path,
                          classification_model: KClassify,
                          media_path: str,
                          localization: List[tator.models.Localization]):
    """
    Predict using a classification model
    :param temp_path: path to the temp directory to store any localizations to later update in the database
    :param classification_model: classification model object
    :param media_path: path to the media file
    :param localization:  Tator localization object
    """
    im = Image.open(media_path)
    # Get the image size
    width, height = im.size
    for loc in localization:
        x = int(loc.x * width)
        y = int(loc.y * height)
        w = int(loc.width * width)
        h = int(loc.height * height)
        crop = im.crop((x, y, x + w, y + h))

        # Convert crop to bytes
        byte_io = io.BytesIO()
        crop.save(byte_io, format='JPEG')
        byte_io.seek(0)
        # Classify the crops
        threshold = 0.3
        predictions = classification_model.predict_bytes(byte_io, top_n=5, threshold=threshold)
        if len(predictions) == 0:
            info(f'No predictions for localization {loc.id} > {threshold}')
            continue
        # Keep the top prediction if there is not a lot of confusion in the top 3
        if len(predictions) > 1:
            if predictions[0]['score'] - predictions[1]['score'] < 0.3:
                warn(f'Confusion in top 2 predictions for localization {loc.id}')
                continue
        if len(predictions) > 2:
            if predictions[1]['score'] - predictions[2]['score'] < 0.3:
                warn(f'Confusion in top 3 predictions for localization {loc.id}')
                continue

        results = {'prediction': predictions[0], 'localization': loc}
        temp_path = temp_path / f'{loc.id}.pkl'
        with temp_path.open('wb') as f:
            pickle.dump(results, f)


def classify(api: tator.api,
             project_id: int,
             group: str,
             version: str,
             generator: str,
             output_path: Path,
             model_url: str):
    """
    Assign localizations to a class based on a classification model. Assumes media is downloaded to the local machine
     in the expected format in output_path with the download command.
    :param api: tator.api
    :param project_id: project id
    :param group: group name to assign classifications to
    :param version: version tag to assign classifications to
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param output_path: path to the output directory data was downloaded to
    :param model_url: fastai model url
    """

    classification_model = KClassify(model_url)

    info(f'Testing model {model_url}')
    test_image = f'{Path(__file__).parent.parent.absolute()}/tests/data/goldfish.jpg'
    results = classification_model.predict_file(test_image, top_n=1, threshold=0.0)
    if len(results) == 0:
        err('Model did not return any predictions')
        exit(-1)

    info('Model OK')

    if generator:
        attribute_filter = [f"generator::{generator}"]
    if group:
        attribute_filter += [f"group::{group}"]
    if version:
        attribute_filter += [f"version::{version}"]

    attribute_filter += ['Label::Unknown']

    num_records = api.get_localization_count(project=project_id,
                                             attribute=attribute_filter)

    if num_records == 0:
        info(f'No localizations to classify found in project {project_id} with filter {attribute_filter}')
        return

    info(f'Found {num_records} localizations to classify')

    # Grab all localizations for the media ids that are 'Unknown', 500 at a time, classify and load them
    inc = min(500, num_records)
    for start in range(0, num_records, inc):
        info(f'Query records {start} to {start + 500}')
        localizations = api.get_localization_list(project=project_id,
                                                  attribute=attribute_filter,
                                                  start=start,
                                                  stop=start + 500)
        if len(localizations) == 0:
            break

        info(f'Found {len(localizations)} localizations to classify')

        # Create a dictionary of media ids to localizations
        media_to_localizations = {}
        for loc in localizations:
            if loc.media not in media_to_localizations:
                # Get the media object using the media name
                media = api.get_media(loc.media)

                # Check if the media is downloaded to the output path
                media_path = output_path / version / 'images' / media.name

                if not media_path.exists():
                    info(f'Media {media_path} not found.')
                    continue

                media_to_localizations[media_path.as_posix()] = []
            media_to_localizations[media_path.as_posix()].append(loc)

        # If no media is found, exit
        if len(media_to_localizations) == 0:
            info('No media found')
            return

        # For each media id, run the classification model in parallel with model.predict
        info('Running classification model')

        # Create a multiprocessing pool
        num_processes = multiprocessing.cpu_count()

        # if the number of me is less than the number of processes, use the number of media as the number of processes
        if len(media_to_localizations) < num_processes:
            num_processes = len(media_to_localizations)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Single thread example for testing
            for media_path, localizations in media_to_localizations.items():
                predict_classify_top1(temp_path, classification_model, media_path, localizations)

            '''with multiprocessing.Pool(num_processes) as pool:
                a = [(temp_path, classification_model, media_path, localizations) for media_path, localizations in media_to_localizations.items()]
                pool.starmap(predict_classify_top1, a)'''

            info('Writing localizations back to the server')
            for l in temp_path.glob('*.pkl'):
                with l.open('rb') as f:
                    results = pickle.load(f)
                    loc = results['localization']
                    pred = results['prediction']

                    # Replace machine name with actual name e.g. sp. A sp__A or krill_molt to krill molt
                    class_name = pred['class'].replace('__', '. ')
                    class_name = class_name.replace('_', ' ')
                    loc.attributes['Label'] = class_name
                    loc.attributes['concept'] = class_name
                    loc.attributes['score'] = pred['score']
                    loc.attributes['group'] = 'VERIFY'
                    loc.version = None

                    # Update the localization
                    info(loc)
                    info(f'Loading localization id {loc.id} {loc.attributes["concept"]} {loc.attributes["score"]}')
                    api.update_localization(loc.id, loc)


def assign_cluster(api: tator.api,
                   project_id: int,
                   group: str,
                   version: str,
                   generator: str,
                   clusters: [],
                   concept: str,
                   label: str):
    """
    Assign a cluster a new concept
    :param api: tator.api
    :param project_id: project id
    :param group: group name
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param clusters: list of clusters to reassign,
    :param concept: concept to assign to the cluster
    :param label: label to assign to the cluster
    """
    # Require either a concept or a label
    if not concept and not label:
        exception('Must provide either a concept or a label')
        exit(-1)

    # Fetch localizations in the cluster and update them up to 500 at a time
    num_assigned = 0
    for c in clusters:
        cluster_filter = f"cluster::{c.strip()}"

        if generator:
            attribute_filter = [f"generator::{generator}"]
        if group:
            attribute_filter += [f"group::{group}"]
        if version:
            attribute_filter += [f"version::{version}"]

        attribute_filter += [cluster_filter]
        num_records = api.get_localization_count(project=project_id,
                                                 attribute=attribute_filter)
        inc = min(500, num_records)
        for start in range(0, num_records, inc):
            info(f'Query records {start} to {start + 500}')
            localizations = api.get_localization_list(project=project_id,
                                                      attribute=attribute_filter,
                                                      start=start,
                                                      stop=start + 500)
            # Update the concept and Label attributes
            for l in localizations:
                if concept:
                    l.attributes['concept'] = concept
                if label:
                    l.attributes['Label'] = concept
                l.version = None

                api.update_localization(l.id, l)
                num_assigned += 1

    info(f'Assigned {num_assigned} records to concept {concept} and label {label}')


def delete(api: tator.api,
           project_id: int,
           group: str,
           version: str,
           generator: str,
           concepts=None,
           labels=None,
           clusters=None,
           dry_run: bool = False):
    """
    Delete by attribute generator, group, version, concept, Label and/or cluster
    :param api: tator.api
    :param project_id: project id
    :param group: group name
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param concepts: list of concepts to delete
    :param labels: list of labels to delete
    :param clusters: list of clusters to delete
    :param dry_run: if True, do not delete, just print
    """

    attribute_filter = []
    if concepts:
        attribute_filter = [f"concept::{concept.strip()}" for concept in concepts]
    if clusters:
        attribute_filter = [f"cluster::{cluster.strip()}" for cluster in clusters]
    if labels:
        attribute_filter = [f"Label::{label.strip()}" for label in labels]
    if generator:
        attribute_filter += [f"generator::{generator}"]
    if group:
        attribute_filter += [f"group::{group}"]
    if version:
        attribute_filter += [f"version::{version}"]

    num_records = api.get_localization_count(project=project_id,
                                             attribute=attribute_filter)

    info(
        f'Found {num_records} localizations to delete in generator {generator} group {group} version {version} concepts {concepts} clusters {clusters}')

    if num_records == 0:
        return

    if dry_run:
        info(f'Dry run, not deleting {num_records} localizations')
        return

    info(f'Deleting {num_records} localizations ...')
    api.delete_localization_list(project=project_id,
                                 attribute=attribute_filter)

    info(f'Deleted {num_records} localizations')


def assign_iou(api: tator.api,
               project_id: int,
               group_source: str,
               group_target: str,
               version: str,
               generator: str,
               conf: float):
    """
    Assign a concepts based from one group to another with a confidence threshold > conf and IOU > 0.2
    :param api: tator.api
    :param project_id: project id
    :param group_source: group name to copy from
    :param group_target: group name to copy to
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param conf: confidence threshold to use to assign concepts
    """

    def get_filter(group):
        attribute_filter = []
        if generator:
            attribute_filter += [f"generator::{generator}"]
        if group:
            attribute_filter += [f"group::{group}"]
        if version:
            attribute_filter += [f"version::{version}"]
        return attribute_filter

    attribute_filter_src = get_filter(group_source)
    attribute_filter_tgt = get_filter(group_target)

    # Get the source localizations
    num_records_source = api.get_localization_count(project=project_id,
                                                    attribute=attribute_filter_src)
    # Get the target localizations
    num_records_target = api.get_localization_count(project=project_id,
                                                    attribute=attribute_filter_tgt)

    def get_records(attribute_filter: [], num_records: int):
        # Fetch localizations up to 500 at a time based on the attribute filter
        inc = min(500, num_records)
        records = []
        for start in range(0, num_records, inc):
            info(f'Query records {start} to {start + 500} for {attribute_filter}')
            r = api.get_localization_list(project=project_id,
                                          attribute=attribute_filter,
                                          start=start,
                                          stop=start + 500)
            records += r

        return records

    records_source = get_records(attribute_filter_src, num_records_source)
    records_target = get_records(attribute_filter_tgt, num_records_target)

    # Create a dictionary of target localizations by media id
    def get_target_by_media(records_target: []):
        """Create a dictionary of source localizations by media id"""
        target_by_media = {}
        for r in records_target:
            if r.media not in target_by_media:
                target_by_media[r.media] = []
            target_by_media[r.media].append(r)

        return target_by_media

    def calculate_iou(box1, box2):
        # box1 and box2 should be in the format (x, y, width, height)
        x1, y1, w1, h1 = box1.x, box1.y, box1.width, box1.height
        x2, y2, w2, h2 = box2.x, box2.y, box2.width, box2.height

        # Calculate the coordinates of the intersection rectangle
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        x_intersection_end = min(x1 + w1, x2 + w2)
        y_intersection_end = min(y1 + h1, y2 + h2)

        # Calculate the intersection area
        intersection_area = max(0, x_intersection_end - x_intersection) * max(0,
                                                                              y_intersection_end - y_intersection)

        # Calculate the union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        # Calculate the IOU
        iou = intersection_area / union_area

        return iou

    target_by_media = get_target_by_media(records_target)
    source_by_media = get_target_by_media(records_source)

    # For each source localization, find the target localization with the highest IOU and keep the source concept if 
    # it score > conf
    for r in records_source:
        if r.media in target_by_media:
            target_localizations = target_by_media[r.media]
            max_iou = 0
            score = r.attributes['score']
            concept = r.attributes['concept']
            if score < conf:
                info(f'Skipping localization {r.id} with score {score}')
                continue

            max_iou_target = None
            for t in target_localizations:
                iou = calculate_iou(r, t)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_target = t

            if max_iou_target:
                # Assign the concept but only if the target localization is Unknown as it may already be assigned
                if max_iou_target.attributes['Label'] != 'Unknown' or max_iou_target.attributes['concept'] != 'Unknown':
                    info(f"Skipping as {max_iou_target.id} already assigned to {max_iou_target.attributes['Label']}")
                    continue

                info(f"Assigning concept {concept} with confidence {score} to {max_iou_target.id} in media {r.media}")
                max_iou_target.attributes['concept'] = concept
                max_iou_target.attributes['Label'] = concept
                max_iou_target.attributes['score'] = score
                max_iou_target.x = r.x
                max_iou_target.y = r.y
                max_iou_target.width = r.width
                max_iou_target.height = r.height
                max_iou_target.version = None
                api.update_localization(max_iou_target.id, max_iou_target)

                target_localizations.remove(max_iou_target)
                source_by_media[r.media] = target_localizations


def assign_nms(api: tator.api,
               project_id: int,
               group: str,
               version: str,
               exclude: [],
               include: [],
               min_iou: float = 0.5,
               min_score: float = 0.2,
               dry_run: bool = False):
    """
    Assign the best concepts in all groups for any image to the new group using NMS.
    :param api: tator.api
    :param project_id: project id
    :param group: group name to assign boxes output from nms to
    :param exclude: (optional) list of concepts/Labels to exclude
    :param include: (optional) list of concepts/Labels to include, default to all if not specified
    :param version: version tag
    :param min_iou: (optional) minimum iou to filter localizations
    :param min_score: (optional) minimum score to filter localizations
    :param dry_run: (optional) if True, do not create any localizations
    """

    # Get the localization type
    localization_types = api.get_localization_type_list(project_id)

    # the box type is the one with the name 'Boxes'
    box_type = None
    for l in localization_types:
        if l.name == 'Boxes':
            box_type = l.id
            break

    if group is None:
        err(f'Group must be specified')
        return

    # Fail if we could not find the box type
    if box_type is None:
        err(f'Could not find localization type "Boxes"')
        return

    attribute_filter = []
    if version:
        attribute_filter.append([f"version::{version}"])
    if include:
        for i in include:
            attribute_filter += [f"concept::{i}"]

    # Check if any records exist with the group first, and alert the user if so
    num_in_group = api.get_localization_count(project=project_id, attribute=[f"group::{group}"])
    if num_in_group > 0:
        info(f"Found {num_in_group} records in group {group}. "
             f"Please remove them before running this script.")
        return

    num_records = api.get_localization_count(project=project_id, attribute=attribute_filter)

    if num_records == 0:
        info(f"No records found with version {version} "
             f"including {include if include else 'everything'} ")
        return

    if dry_run:
        info(f"Dry run. Found {num_records} records with version {version} and group {group} "
             f"including {include if include else 'everything'} ")
        return

    # Fetch localizations up to 2000 at a time based on the attribute filter
    inc = min(2000, num_records)
    records = []
    for start in range(0, num_records, inc):
        info(f'Query records {start} to {start + 2000}')
        localizations = api.get_localization_list(project=project_id,
                                                  attribute=attribute_filter,
                                                  start=start,
                                                  stop=start + 2000)

        if len(localizations) == 0:
            break

        # Remove localizations if exclude is set
        if exclude:
            new_localizations = [l for l in localizations if
                                 l.attributes['concept'] not in exclude and
                                 l.attributes['Label'] not in exclude]
        else:
            new_localizations = localizations

        records += new_localizations

    info(f"Found {num_records} records with version {version} and group {group} "
         f"including {include if include else 'everything'} "
         f"excluding {exclude if exclude else 'none'}")

    # Create a dictionary of target localizations by media id
    by_media = {}
    for r in records:
        if r.media not in by_media:
            by_media[r.media] = []
        by_media[r.media].append(r)

    # Iterate over the dictionary of localizations by media
    # For each media, run NMS to get the best boxes
    num_created = 0
    for media, localizations in by_media.items():

        # Create a tensor of scores and boxes
        scores = torch.tensor([l.attributes['score'] for l in localizations], dtype=torch.float32)
        boxes_nms = torch.tensor([[l.x, l.y, l.x + l.width, l.y + l.height] for l in localizations],
                                 dtype=torch.float32)  # to run nms x1, y1, x2, y2
        boxes = [[l.x, l.y, l.width, l.height] for l in localizations]  # tator API needs x, y, width, height
        # Add random noise to the scores to break ties
        scores += torch.rand(scores.shape) * 1e-3
        boxes_nms += torch.rand(boxes_nms.shape) * 1e-3
        labels = [l.attributes['Label'] for l in localizations]
        concepts = [l.attributes['concept'] for l in localizations]
        iou_threshold = min_iou

        selected_indices = nms(boxes_nms, scores, iou_threshold)

        # Convert tensors to lists
        selected_indices = selected_indices.tolist()
        scores = scores.tolist()

        # Clamp scores to 1.0
        scores = [min(s, 1.0) for s in scores]

        selected_scores = [scores[i] for i in selected_indices]
        selected_boxes = [boxes[i] for i in selected_indices]
        selected_labels = [labels[i] for i in selected_indices]
        selected_concepts = [concepts[i] for i in selected_indices]

        # Create a list of boxes for the API using the selected boxes, scores, and classes
        boxes = []
        for box, score, label, concept in zip(selected_boxes, selected_scores, selected_labels, selected_concepts):
            if score < min_score:
                continue
            new_box = gen_box_localization(box_type, box, score, media, project_id, concept, label, group)
            boxes.append(new_box)

        # Create the localizations 200 at a time
        for b in range(0, len(boxes), 200):
            info(f'Creating {len(boxes[b:b + 200])} localizations for media {media} to group {group}...')
            info(boxes[b:b + 200])
            response = api.create_localization_list(project_id, boxes[b:b + 200])
            debug(response)
            num_created += len(boxes[b:b + 200])

    info(f'Created {num_created} total localizations in group {group}...')


def gen_box_localization(box_type: int,
                         box: [float],
                         score: float,
                         media_id: int,
                         project_id: int,
                         concept: str,
                         label: str,
                         group: str):
    """
    Generate a localization spec for a box
    :param box_type: The localization type ID for the box type
    :param box:  [x, y, w, h]
    :param score: score of the localization
    :param media_id: The Tator media ID.
    :param project_id: The Tator project ID.
    :param concept: Concept to assign boxes output to
    :param label: Label to assign boxes output to
    :param group: group name to assign boxes output to
    :return: A localization spec
    """
    attributes = {
        'concept': concept,
        'Label': label,
        'score': score,
        'group': group,
    }

    out = {
        'type': box_type,
        'media_id': media_id,
        'project': project_id,
        'x': box[0],
        'y': box[1],
        'width': box[2],
        'height': box[3],
        'frame': 0,
        'attributes': attributes
    }
    return {**out}
