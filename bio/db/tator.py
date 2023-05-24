import os
from pathlib import Path

import tator

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


def download_data(api: tator.api, version: str, output_path:Path):
    """
    Download a dataset based on a version tag for training
    :param api: tator.api
    :param version: version tag
    :param output_path: output directory to save the dataset
    """
    try:
        # Get the version
        version = api.get_version_list(version_tag=version)[0]
        info(version)

        # Get the dataset
        dataset = api.get_dataset(version.dataset)
        info(dataset)

        # Get the media
        media = api.get_media_list(dataset=dataset.id)
        info(media)

        # Download the media
        for m in media:
            info(f'Downloading {m.name}')
            api.download_media(m.id, output_path.as_posix())

    except Exception as e:
        exception(e)
        exit(-1)

