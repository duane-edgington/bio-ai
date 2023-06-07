import multiprocessing
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import tator
import tensorflow as tf



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

    loc_array = []

    for l in localizations:
        # Crop the image
        cropped_image = image.crop((l.x, l.y, l.x + l.width, l.y + l.height))

        # Resize the image
        resized_image = cropped_image.resize(image_size)

        loc_array.append(l.attributes['Label'])

    # Convert to numpy array
    image_array = np.asarray(resized_image)

    # Save the image and label to the temporary directory as npy files
    np.save(temp_path / f'{image_path.stem}.npy', image_array)
    np.save(temp_path / f'{image_path.stem}_label.npy', loc_array)


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
        for image_path in temp_path.glob('*.npy'):
            if 'label' in image_path.name:
                continue
            images.append(np.load(image_path.as_posix()).astype('float32')/255.0)
            for l in np.load(temp_path / f'{image_path.stem}_label.npy'):
                labels.append(class_names.index(l))

        # One-hot encode the labels
        labels = tf.keras.utils.to_categorical(labels, len(class_names))

        # Save the data
        np.save(data_path / 'images.npy', images)
        np.save(data_path / 'labels.npy', labels)

    return images, labels


def download_data(api: tator.api, project_id: int, version: str, generator: str, output_path: Path, concept_list: [],
                  cifar: bool = False):
    """
    Download a dataset based on a version tag for training
    :param api: tator.api
    :param project_id: project id
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

        # Get the annotations in chunks of 500 or less if there are less than 500
        num_records = api.get_localization_count(project=project_id,
                                                 attribute=[f"generator::{generator}"])

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
            info(f'Query records {start} to {inc}')
            new_localizations = api.get_localization_list(project=project_id,
                                                          attribute=[f"generator::{generator}"],
                                                          start=start,
                                                          stop=start + 500)
            if len(new_localizations) == 0:
                break

            # Filter out localizations that are not in the concept list, or skip if the list is "all"
            if concept_list != "all":
                new_localizations = [l for l in new_localizations if l.attributes['concept'] in concept_list]

            if len(new_localizations) == 0:
                continue

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
                    f.write(f'{label_idx} {loc.x} {loc.y} {loc.width} {loc.height}\n')
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
                    print(f"Download progress: {progress}%")

        # optionally create a CIFAR dataset
        if cifar:
            info(f'Creating output directory {output_path} in CIFAR format')
            cifar_path = output_path / 'cifar'
            cifar_path.mkdir(exist_ok=True)
            # Create the CIFAR dataset
            images, labels = create_cifar_dataset(cifar_path, media_lookup_by_id,  localizations, labels)
            np.save(cifar_path / 'images.npy', images)
            np.save(cifar_path / 'labels.npy', labels)
    except Exception as e:
        exception(e)
        exit(-1)
