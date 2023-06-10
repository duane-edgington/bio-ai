import multiprocessing
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import tator
import tensorflow as tf
import itertools


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


def get_image_label(temp_path: Path, image_path: Path, localizations):
    """
    Get the image and label for a localization
    :param temp_path: Path to the temporary directory
    :param image_path: Path to the image
    :param localizations: Bounding box localization
    :return: image, label
    """
    image_size = (32, 32)

    # Get the image
    image = Image.open(image_path)

    width, height = image.size

    for i, l in enumerate(localizations):
        # Crop the image
        cropped_image = image.crop((int(width*l.x),
                                    int(height*l.y),
                                    int(width*(l.x + l.width)),
                                    int(height*(l.y + l.height))))

        # Resize the image
        resized_image = cropped_image.resize(image_size)

        # Convert to numpy array
        image_array = np.asarray(resized_image)

        # Save the image and label to the temporary directory as npy files
        np.save(temp_path / f"{image_path.stem}-image-{l.attributes['Label']}.npy", image_array)


def create_cifar_dataset(data_path: Path, media_lookup_by_id, localizations: [], class_names: []):
    """
    Create CIFAR formatted data from a list of media and localizations
    :param data_path: Path to save the data
    :param media_lookup_by_id: Media id to media path lookup
    :param localizations: List of localizations
    :param class_names: List of class names
    """
    images = []
    labels = []

    with tempfile.TemporaryDirectory() as temp_path:
        temp_path = Path(temp_path)

        # Crop the images in parallel using multiprocessing to speed up the processing
        num_processes = min(multiprocessing.cpu_count(), len(media_lookup_by_id))
        with multiprocessing.Pool(num_processes) as pool:
            args = [[temp_path, Path(media_path), [l for l in localizations if l.media == media_id]]  for media_id, media_path in media_lookup_by_id.items()]
            pool.starmap(get_image_label, args)

        # Read in the images and labels from a temporary directory
        for npy in sorted(temp_path.glob('*.npy')):
            images.append(np.load(npy.as_posix()).astype('int32'))

            # label name is the last part of the filename after the -
            label_name = npy.stem.split('-')[-1]
            labels.append([int(class_names.index(label_name))])

        # Save the data
        image_path = data_path / 'images.npy'
        label_path = data_path / 'labels.npy'
        if image_path.exists():
            image_path.unlink()
        if label_path.exists():
            label_path.unlink()
        np.save(data_path / 'images.npy', images)
        np.save(data_path / 'labels.npy', labels)

    return images, labels


def download_data(api: tator.api, project_id: int, group:str, version: str, generator: str, output_path: Path, concept_list: [],
                  cifar: bool = False):
    """
    Download a dataset based on a version tag for training
    :param api: tator.api
    :param project_id: project id
    :param group: group name
    :param version: version tag
    :param generator: generator name, e.g. 'vars-labelbot' or 'vars-annotation'
    :param output_path: output directory to save the dataset
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

        
        # Get the annotations in chunks of 500 or less if there are less than 500
        if concept_list!=['all']:
            # Fetch number of localizations for user requested concepts only 
            # api only allows us to fetch one concept at a time
            num_records_per_concept = [api.get_localization_count(project=project_id,
                                                 attribute=attribute_filter + [f"concept::{concept.strip()}"]) for concept in concept_list]
            num_records = sum(num_records_per_concept)

        else:
            # Get all the localizations avaiable in database
            num_records = api.get_localization_count(project=project_id,
                                                 attribute=attribute_filter)

        info(f'Found {num_records} records for version {version.name} and generator {generator}')

        if num_records == 0:
            err(f'Could not find any records for version {version.name} and generator {generator}')
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

            if concept_list!=['all']:
                # Fetch localizations for user requested concepts only 
                # api only allows us to fetch one concept at a time
                new_localizations_per_concept = [api.get_localization_list(project=project_id,
                                                     attribute=attribute_filter + [f"concept::{concept.strip()}"], start=start, 
                                                     stop=start + 500) for concept in concept_list]
                # new_localizations_per_concept is a list of lists. Merge into a single list.
                new_localizations = list(itertools.chain.from_iterable(new_localizations_per_concept))

            else:
                new_localizations = api.get_localization_list(project=project_id,
                                                          attribute=attribute_filter,
                                                          start=start,
                                                          stop=start + 500)
            if len(new_localizations) == 0:
                break

            # Filter out localizations that are not in the concept list, or skip if the list is "all"
            # if concept_list != ["all"]:
            #     new_localizations = [l for l in new_localizations if l.attributes['concept'].strip().lower() in concept_list]

            #if len(new_localizations) == 0:
            #    break

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
                    #f.write(f'{label_idx} {loc.x} {loc.y} {loc.width} {loc.height}\n')
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

            images, labels = create_cifar_dataset(cifar_path, media_lookup_by_id,  localizations, labels)
            np.save(cifar_path / 'images.npy', images)
            np.save(cifar_path / 'labels.npy', labels)
    except Exception as e:
        exception(e)
        exit(-1)
