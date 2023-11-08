# bio-ai, Apache-2.0 license
# Filename: bio/db/ml.py
# Description: handles calling various ml algorithms, e.g. combining models using NMS, classification, etc.
# and loads the results back into the database

import io
import multiprocessing
import pickle
import tempfile
import requests
from pathlib import Path
from typing import List

import tator
import torch
from PIL import Image
from torchvision.ops import nms

from bio.db.tator_db import gen_localization
from bio.db.url_utils import is_valid_url, extract_image_links
from bio.logger import info, warn, err, exception, debug
from bio.model import KClassify, YOLOv5


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


def detect(api: tator.api,
           box_type: int,
           project_id: int,
           image_type: int,
           group: str,
           version: str,
           generator: str,
           base_image_url: str,
           model_url: str):
    """
    Assign localizations to a class based on a detection model.
    :param api: tator.api
    :param project_id: project id
    :param image_type: image type id
    :param group: group name to assign classifications to
    :param version: version tag to assign classifications to
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param base_image_url: root url images are located at
    :param model_url: fastai model url
    """

    detection_model = YOLOv5(model_url)

    # Extract image links
    image_urls = extract_image_links(base_image_url)
    if not image_urls:
        err(f'Could not extract image links from {base_image_url}')
        return

    for image_url in image_urls:

        if not is_valid_url(image_url):
            err(f'Could not find image {image_url}')
            return

        # Download the media to a temp directory
        with tempfile.TemporaryDirectory() as temp_dir:

            # Download the media
            media_path = Path(temp_dir) / 'media.jpg'
            with media_path.open('wb') as f:
                f.write(requests.get(image_url).content)

            # Get the image size
            im = Image.open(media_path)
            width, height = im.size

            # Run the detection model on the media
            detections = detection_model.predict_file(media_path, threshold=0.1)

            # Load the results
            if len(detections) == 0:
                info(f'No detections found for {image_url}')
                return

            # upload the image url reference
            info(f'Importing image url {image_url}')
            error = False
            try:
                for progress, response in tator.util.import_media(api,
                                                                  image_type,
                                                                  image_url,
                                                                  attributes={},
                                                                  fname=Path(image_url).name):
                    info(f"Creating progress: {progress}%")
                    info(f'Uploading {image_url}')
            except Exception as e:
                err(f'Error: {e}')
                if 'already exists' in str(e).lower():
                    info(f'Image {image_url} already exists')
                if 'not found' in str(e).lower():
                    err(f'Image {image_url} not found')
                    error = True

            if error:
                return

            # Get the media id
            media = api.get_media_list(project=project_id, name=Path(image_url).name)

            if len(media) == 0:
                err(f'Could not find media {image_url}')
                return

            media = media[0]

            # Load the detections
            info(f'Loading {len(detections)} detections')
            locs = []
            for d in detections:
                attributes = {
                    'Label': d['class_name'],
                    'score': d['confidence'],
                    'concept': d['class_name'],
                    'generator': generator,
                    'group': group,
                    'cluster': -1
                }

                out = {
                    'type': box_type,
                    'media_id': media.id,
                    'project': project_id,
                    'x': d['x'] / width,
                    'y': d['y'] / height,
                    'width': d['width'] / width,
                    'height': d['height'] / height,
                    'frame': 0,
                    'attributes': attributes,
                }

                locs.append(out)
                info(f'Loading localization {out}')
            if len(locs) > 0:
                api.create_localization_list(project_id, locs)

        info(f'Loaded {len(detections)} detections')


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
    :param group: (optional) group name to query for localizations, default to all groups if not specified
    :param exclude: (optional) list of concepts/Labels to exclude
    :param include: (optional) list of concepts/Labels to include, default to all if not specified
    :param version: version tag
    :param min_iou: (optional) minimum iou to filter localizations
    :param min_score: (optional) minimum score to filter localizations
    :param dry_run: (optional) if True, do not create any localizations
    """
    group_assign = "NMS"

    # Get the localization type
    localization_types = api.get_localization_type_list(project_id)

    # the box type is the one with the name 'Boxes'
    box_type = None
    for l in localization_types:
        if l.name == 'Boxes':
            box_type = l.id
            break

    # Fail if we could not find the box type
    if box_type is None:
        err(f'Could not find localization type "Boxes"')
        return

    # Fail if iou_threshold is not between 0 and 1 or it is None
    if min_iou is None or min_iou < 0 or min_iou > 1:
        err(f'Invalid iou_threshold {min_iou}')
        return

    # Fail if score_threshold is not between 0 and 1 or it is None
    if min_score is None or min_score < 0 or min_score > 1:
        err(f'Invalid score_threshold {min_score}')
        return

    attribute_filter = []
    if version:
        attribute_filter.append(f"version::{version}")
    if group:
        attribute_filter.append(f"group::{group}")

    # Check if any records exist with the group first, and alert the user if so
    num_in_group = api.get_localization_count(project=project_id, attribute=[f"group::{group_assign}"])
    if num_in_group > 0:
        info(f"Found {num_in_group} records in group {group_assign}. "
             f"Please remove them before running this script.")
        return

    num_records = 0
    if include:
        for i in include:
            num_records += api.get_localization_count(project=project_id,
                                                      attribute=attribute_filter + [f"concept::{i}"])
    else:
        num_records = api.get_localization_count(project=project_id, attribute=attribute_filter)

    if num_records == 0:
        info(f"No records found with version {version} and group {group} "
             f"including {include if include else 'everything'} ")
        return

    # Fetch localizations up to 5000 at a time based on the attribute filter
    inc = min(5000, num_records)
    records = []
    if include:
        for i in include:
            info(f'Query records for {i}')
            new_records = api.get_localization_list(project=project_id,
                                                    attribute=attribute_filter + [f"concept::{i}"],
                                                    start=0,
                                                    stop=num_records)
            if len(new_records) == 0:
                break
            records += new_records
    else:
        for start in range(0, num_records, inc):
            info(f'Query records {start} to {start + 5000}')
            new_records = api.get_localization_list(project=project_id,
                                                    attribute=attribute_filter,
                                                    start=start,
                                                    stop=start + 5000)
            if len(new_records) == 0:
                break
            records += new_records

        # Only keep localizations in the group MERGE or MERGE_CLASSIFY
        records = [l for l in records if l.attributes['group'] in ['MERGE', 'MERGE_CLASSIFY']]

        # Remove localizations if exclude is set
        if exclude:
            filtered_records = [l for l in records if
                                l.attributes['concept'] not in exclude and
                                l.attributes['Label'] not in exclude]
        else:
            filtered_records = records

        num_records = len(filtered_records)

    info(f"Found {num_records} records with version {version} and group {group} "
         f"including {include if include else 'everything'} "
         f"excluding {exclude if exclude else 'none'}")

    # Create a dictionary of target localizations by media id
    by_media = {}
    for r in filtered_records:
        if r.media not in by_media:
            by_media[r.media] = []
        by_media[r.media].append(r)

    # Iterate over the dictionary of localizations by media
    # For each media, run NMS to get the best boxes
    num_created = 0
    num_unknown = 0
    num_revisit = 0
    num_media = 0
    for media, localizations in by_media.items():

        # if any localizations are in the group MERGE_CLASSIFY, then skip this media
        if any([l.attributes['group'] == 'MERGE_CLASSIFY' for l in localizations]):
            info(f"Skipping media {media} because it has localizations in group MERGE_CLASSIFY")
            continue

        num_media += 1
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

        selected_indices = nms(boxes_nms, scores, min_iou)

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
            new_box = gen_localization(box_type, box, score, media, project_id, concept, label, group_assign)
            boxes.append(new_box)

        # Create the localizations 5000 at a time
        for b in range(0, len(boxes), 5000):
            info(f'Creating {len(boxes[b:b + 5000])} localizations for media {media} to group {group_assign}...')
            debug(f'First box: {boxes[b]}')

            num_created += len(boxes[b:b + 5000])
            # Count the number of localizations that were Unknown
            for box in boxes[b:b + 5000]:
                if box['attributes']['Label'] == "Unknown":
                    num_unknown += 1
                if box['attributes']['Label'] == "Revisit":
                    num_revisit += 1
            if dry_run:
                continue

            response = api.create_localization_list(project_id, boxes[b:b + 5000])
            debug(response)

    info(
        f'Created {num_created} total localizations out of {num_records} in {num_media} frames in group {group_assign}. Total Unknown: {num_unknown}. Total Revisit: {num_revisit}')
