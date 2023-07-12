import os
import time
from pathlib import Path
import numpy as np
import tator
import itertools
import torch
from torchvision.ops import nms

from bio.data.generator import create_cifar_dataset
from bio.logger import info, debug, err, exception


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
                  cifar: bool = False):
    """
    Download a dataset based on a version tag for training
    :param api: tator.api
    :param project_id: project id
    :param group: group name
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param output_path: output directory to save the dataset
    :param concept_list: list of concepts to download
    :param cifar: True if the dataset should also be stored in CIFAR format
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
        if generator:
            attribute_filter = [f"generator::{generator}"]
        if group:
            attribute_filter += [f"group::{group}"]
        if concept_list:
            attribute_filter += [f"concept::{concept.strip()}" for concept in concept_list]

        num_records = api.get_localization_count(project=project_id, attribute=attribute_filter)

        info(f'Found {num_records} records for version {version.name} and generator {generator} and group {group}')

        if num_records == 0:
            err(f'Could not find any records for version {version.name} and generator {generator} and group {group}')
            return

        # Create the output directory in the expected format that deepsea-ai expects for training
        # See https://docs.mbari.org/deepsea-ai/data/ for more information
        label_path = output_path / 'labels'
        label_path.mkdir(exist_ok=True)
        media_path = output_path / 'images'
        media_path.mkdir(exist_ok=True)

        localizations = []
        inc = min(500, num_records)
        for start in range(0, num_records, inc):
            info(f'Query records {start} to {start + 500}')

            if concept_list:
                # Fetch localizations for user requested concepts only 
                # api only allows us to fetch one concept at a time
                new_localizations_per_concept = [api.get_localization_list(project=project_id,
                                                                           attribute=attribute_filter + [
                                                                               f"concept::{concept.strip()}"],
                                                                           start=start,
                                                                           stop=start + 500) for concept in
                                                 concept_list]
                # new_localizations_per_concept is a list of lists. Merge into a single list.
                new_localizations = list(itertools.chain.from_iterable(new_localizations_per_concept))

            else:
                new_localizations = api.get_localization_list(project=project_id,
                                                              attribute=attribute_filter,
                                                              start=start,
                                                              stop=start + 500)

                # Remove localizations that have a conceppt of 'Unknown' or 'Revisit'
                new_localizations = [l for l in new_localizations if
                                     l.attributes['concept'] not in ['Unknown', 'Revisit']]
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
        info(f'Total unique media items are {len(all_media)}')

        # Get all the unique media names
        media_names = list(set([m.name.split('.png')[0] for m in all_media]))

        # Get all the unique Label attributes and sort them alphabetically
        labels = list(sorted(set([l.attributes['Label'] for l in localizations])))

        # Write the labels to a file called label-map.txt
        with (output_path / 'label-map.txt').open('w') as f:
            for label in labels:
                f.write(f'{label}\n')

        # Create YOLO format files, one per media
        for media_name in media_names:
            # Get the media object
            media = [m for m in all_media if m.name.split('.png')[0] == media_name][0]

            media_lookup_by_id[media.id] = media_path / media.name

            with (label_path / f'{media_name}.txt').open('w') as f:
                # Get all the localizations for this media
                media_localizations = [l for l in localizations if l.media == media.id]
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

        for m in media_ids:
            media = api.get_media(m)
            out_path = media_path / media.name
            if not out_path.exists():
                for progress in tator.util.download_media(api, media, out_path):
                    debug(f"{media.name} download progress: {progress}%")

        # optionally create a CIFAR formatted dataset
        if cifar:
            info(f'Creating output directory {output_path} in CIFAR format')
            cifar_path = output_path / 'cifar'
            cifar_path.mkdir(exist_ok=True)

            images, labels = create_cifar_dataset(cifar_path, media_lookup_by_id, localizations, labels)
            np.save(cifar_path / 'images.npy', images)
            np.save(cifar_path / 'labels.npy', labels)
    except Exception as e:
        exception(e)
        exit(-1)


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
           concepts: [],
           labels: [],
           clusters: [],
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

    # Fetch localizations in the cluster and delete them up to 500 at a time
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

    num_deleted = 0
    inc = min(500, num_records)
    for start in range(0, num_records, inc):
        info(f'Query records {start} to {start + 100}')
        localizations = api.get_localization_list(project=project_id,
                                                  attribute=attribute_filter,
                                                  start=start,
                                                  stop=start + 100)

        info(f'Deleting {len(localizations)} localizations ...')
        if localizations:
            for l in localizations:
                num_deleted += 1
                api.delete_localization(l.id)

        # Wait a bit to avoid rate limiting
        info(f'Waiting 5 seconds to avoid rate limiting')
        time.sleep(5)

    info(f'Deleted {num_deleted} localizations')


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
               dry_run: bool = False):
    """
    Assign the best concepts in all groups for any image to the new group using NMS.
    :param api: tator.api
    :param project_id: project id
    :param group: group name to assign boxes output from nms to
    :param exclude: (optional) list of concepts/Labels to exclude
    :param include: (optional) list of concepts/Labels to include, default to all if not specified
    :param version: version tag
    :param dry_run: (optional) if True, do not update any localizations
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
            attribute_filter += [f"Label::{i}"]

    num_records = api.get_localization_count(project=project_id, attribute=attribute_filter)

    if num_records == 0:
        info(f"No records found with version {version} and group {group} "
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
        iou_threshold = 0.5

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
        # Ignore any boxes with a score less than 0.2 - these are likely false positives
        boxes = []
        for box, score, label, concept in zip(selected_boxes, selected_scores, selected_labels, selected_concepts):
            if score < 0.2:
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
