# bio-ai, Apache-2.0 license
# Filename: bio/db/tator_db.py
# Description: handles loading media and localizations results into the database

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import tator
from pascal_voc_writer import Writer

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
                  cifar_size: int = 32,
                  skip_image_download: bool = False,
                  save_score: bool = False,
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
    :param skip_image_download: (optional) True if the images should not be downloaded
    :param save_score: (optional) True if the score should be saved in the YOLO format
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
        attribute_filter = None
        num_records = 0
        if generator:
            attribute_filter = [f"generator::{generator}"]
        if group:
            if attribute_filter:
                attribute_filter += [f"group::{group}"]
            else:
                attribute_filter = [f"group::{group}"]
        if concept_list:
            for c in concept_list:
                num_records += api.get_localization_count(project=project_id, attribute_contains=attribute_filter + [f"concept::{c}"])
        else:
            num_records = api.get_localization_count(project=project_id, attribute_contains=attribute_filter)

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
            info(f'Creating VOC files in {voc_path}')
        if coco:
            coco_path = output_path / 'coco'
            coco_path.mkdir(exist_ok=True)
            info(f'Creating COCO files in {coco_path}')

        localizations = []
        inc = min(5000, num_records)
        if concept_list:
            for c in concept_list:
                for start in range(0, num_records, inc):
                    filter = attribute_filter + [f"concept::{c}"]
                    info(f'Query records {start} to {start + inc} using attribute filter {filter} ')

                    new_localizations = api.get_localization_list(project=project_id,
                                                                  attribute_contains=filter,
                                                                  start=start,
                                                                  stop=start + 5000)

                    if len(new_localizations) == 0:
                        break

                    localizations = localizations + new_localizations
        else:
            for start in range(0, num_records, inc):
                info(f'Query records {start} to {start + inc} using attribute filter {attribute_filter}')

                new_localizations = api.get_localization_list(project=project_id,
                                                              attribute_contains=attribute_filter,
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
        def get_media_stem(media_path: Path) -> str:
            parts = media_path.stem.rsplit('.', 1)
            return '.'.join(parts)

        media_names = list(set([get_media_stem(m) for m in all_media]))

        # Get all the unique Label attributes and sort them alphabetically
        labels = list(sorted(set([l.attributes['Label'] for l in localizations])))

        # Write the labels to a file called labels.txt
        with (output_path / 'labels.txt').open('w') as f:
            for label in labels:
                f.write(f'{label}\n')

        if not skip_image_download:
            # Download all the media files - this needs to be done before we can create the VOC/CIFAR files which reference the
            # media file size
            for media in all_media:
                out_path = media_path / media.name
                if not out_path.exists():
                    info(f'Downloading {media.name} to {out_path}')
                    num_tries = 0
                    success = False
                    while num_tries < 3 and not success:
                        try:
                            for progress in tator.util.download_media(api, media, out_path):
                                debug(f"{media.name} download progress: {progress}%")
                            success = True
                        except Exception as e:
                            err(e)
                            num_tries += 1
                    if num_tries == 3:
                        err(f'Could not download {media.name}')
                        exit(-1)

        # Create YOLO, and optionally COCO, CIFAR, or VOC formatted files
        info(f'Creating YOLO files in {label_path}')
        json_content = {}

        for media_name in media_names:

            # Get the media object
            media = [m for m in all_media if get_media_stem(m) == media_name][0]

            # Get all the localizations for this media
            media_localizations = [l for l in localizations if l.media == media.id]

            media_lookup_by_id[media.id] = media_path / media.name
            yolo_path = label_path / f'{media_name}.txt'
            image_path = media_path / media.name

            # Get the image size from the image using PIL
            image = Image.open(image_path)
            image_width, image_height = image.size

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
                    if save_score:
                        f.write(f"{label_idx} {x} {y} {w} {h} {loc.attributes['score']}\n")
                    else:
                        f.write(f'{label_idx} {x} {y} {w} {h}\n')

            # optionally create VOC files
            if voc:
                # Paths to the VOC file and the image
                voc_xml_path = voc_path / f'{media_name}.xml'
                image_path = (media_path / media.name).as_posix()

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


def gen_localization(box_type: int,
                     box: [float],
                     score: float,
                     media_id: int,
                     project_id: int,
                     concept: str,
                     cluster_id: int,
                     obs_uuid: str,
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
    :param cluster_id: Cluster ID to assign boxes output to
    :param obs_uuid: Observation UUID to assign boxes output to
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
    # Add cluster if it exists
    if cluster_id:
        attributes['cluster'] = str(cluster_id)
    else:
        attributes['cluster'] = "-1"

    # Add obs_uuid if it exists
    if 'obs_uuid':
        attributes['obs_uuid'] = obs_uuid

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


def create_media(
        api,
        media_url: str,
        type_id: int = 1,
        section: str = "All Media"):
    """
    Create a media object in Tator for a media served from a url.
    :param api: The Tator API object.
    :param media_url:  The URL of the media to reference.
    :param type_id: The media type ID. 1 is image, 2 is video.
    :param section: The section to assign to the media - this corresponds to a Tator section which is a collection, akin to a folder.
    :return: The media ID of the created media object.
    """
    f_path = Path(media_url)
    try:

        for progress, response in tator.util.import_media(api,
                                           type_id,
                                           media_url,
                                           section=section,
                                           fname=f_path.name):
            info(f"Creating progress: {progress}%")
            debug(response.message)
            return response.id
    except Exception as e:
        if 'object has no attribute' not in str(e):
            info(f"Error uploading {f_path}: {e}")
            raise e
