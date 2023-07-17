import multiprocessing
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


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
            args = [[temp_path, Path(media_path), [l for l in localizations if l.media == media_id]] for
                    media_id, media_path in media_lookup_by_id.items()]
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
        cropped_image = image.crop((int(width * l.x),
                                    int(height * l.y),
                                    int(width * (l.x + l.width)),
                                    int(height * (l.y + l.height))))

        # Resize the image
        resized_image = cropped_image.resize(image_size)

        # Convert to numpy array
        image_array = np.asarray(resized_image)

        # Save the image and label to the temporary directory as npy files
        np.save(temp_path / f"{image_path.stem}-image-{l.attributes['Label']}.npy", image_array)
