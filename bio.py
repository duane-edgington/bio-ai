# bio-ai, Apache-2.0 license
# Filename: bio/bio.py
# Description: run bio-ai commands to run detection/classification models, download data, etc.

import time

from dotenv import load_dotenv
import click as click
from pathlib import Path

from bio import __version__
from bio.db.tator_db import init_api, download_data, find_project, delete
from bio.db.ml import classify, assign_cluster, assign_iou, assign_nms, detect
from bio.logger import create_logger_file, info, err

# Default values
# The base directory is the same directory as this file
DEFAULT_BASE_DIR = Path(__file__).parent.as_posix()

DEFAULT_VERSION = 'Baseline'
DEFAULT_PROJECT = '901103-biodiversity'


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(
    __version__,
    '-V', '--version',
    message=f'%(prog)s, version %(version)s'
)
def cli():
    """
    Utilities to download data and train models for the 901103 BioDiversity Project.
    """
    pass


@cli.command(name="download", help='Download a dataset for training')
@click.option('--base-dir', default=DEFAULT_BASE_DIR, help='Base directory to save all data to.')
@click.option('--group', help='Group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to download. Defaults to {DEFAULT_VERSION}.')
@click.option('--generator', help='Generator name, e.g. vars-labelbot or vars-annotation')
@click.option('--concepts', default='all', help='Comma separated list of concepts to download.')
@click.option('--voc', is_flag=True, help='True if export as VOC dataset, False if not.')
@click.option('--coco', is_flag=True, help='True if export as COCO dataset, False if not.')
@click.option('--cifar', is_flag=True, help='True if export as CIFAR dataset, False if not.')
@click.option('--cifar-size', default=32, help='Size of CIFAR images.')
@click.option('--save-score', is_flag=True, help='True to save score in YOLO output, False if not.')
@click.option('--skip-image-download', is_flag=True,
              help='Skip image download, only download annotations. CIFAR requires images.')
def download(base_dir: str, group: str, version: str, generator: str, concepts: str, voc: bool, cifar: bool, coco: bool,
             cifar_size: int, save_score: bool, skip_image_download: bool):
    create_logger_file(Path.cwd(), 'download')
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    # Download a dataset by its version. The versioining isn't currently fully wired in, but included for future use.
    info(f'Downloading dataset {version}')
    data_path = base_path / version
    data_path.mkdir(exist_ok=True)

    # Convert comma separated list of concepts to a list
    if concepts == 'all':
        concept_list = None
    else:
        concept_list = concepts.split(',')
        concept_list = [l.strip() for l in concept_list]
    download_data(api, project_id=project.id, group=group, version=version, generator=generator,
                  output_path=data_path, concept_list=concept_list, skip_image_download=skip_image_download,
                  save_score=save_score, cifar_size=cifar_size, voc=voc, coco=coco, cifar=cifar)


@cli.command(name="assign", help='Assign concepts and/or labels to clusters')
@click.option('--group', help='Group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--generator', help='Generator name, e.g. vars-labelbot or vars-annotation')
@click.option('--clusters', default='all', help='Comma separated list of clusters to assign.')
@click.option('--concept', type=str, help='Concept to assign')
@click.option('--label', type=str, help='Label to assign')
def assign(group: str, version: str, generator: str, clusters: str, concept: str, label: str):
    create_logger_file(Path.cwd(), 'assign')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    # Find all localizations with the given generator and group and cluster
    cluster_list = clusters.split(',')
    assign_cluster(api, project_id=project.id, version=version, generator=generator, group=group, clusters=cluster_list,
                   concept=concept, label=label)


@cli.command(name="assign-nms", help='Assign concepts and labels using combined models using NMS')
@click.option('--group', help='New group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--exclude', type=str, help='(Optional) comma separated list of concepts to exclude.')
@click.option('--include', type=str, help='(Optional) comma separated list of concepts to include.')
@click.option('--min-iou', type=float, default=0.5,
              help='(Optional)  minimum iou to filter localizations between 0-1. Defaults to 0.5')
@click.option('--min-score', type=float, default=0.2,
              help='(Optional)  minimum score to filter localizations between 0-1. Defaults to 0.2')
@click.option('--dry-run', is_flag=True, help='Dry run, do not delete')
def assignNMS(group: str, version: str, exclude: str, include: str, min_iou: float, min_score: float, dry_run: bool):
    create_logger_file(Path.cwd(), 'assign-nms')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}. Excluding {exclude}')

    # Convert comma separated list of concepts to a list
    if include:
        include = include.split(',')
    if exclude:
        exclude = exclude.split(',')
    assign_nms(api, project_id=project.id, version=version, group=group, exclude=exclude, include=include,
               dry_run=dry_run, min_score=min_score, min_iou=min_iou)


@cli.command(name="delete", help='Delete clusters, concepts or labels')
@click.option('--group', help='Group name, e.g. VB250')
@click.option('--version', help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--generator', help='Generator name, e.g. vars-labelbot or vars-annotation')
@click.option('--clusters', help='Comma separated list of clusters to delete.')
@click.option('--concepts', type=str, help='Comma separated list of concepts to delete')
@click.option('--labels', type=str, help='Comma separated list of labels to delete')
@click.option('--dry-run', is_flag=True, help='Dry run, do not delete')
def delete_bulk(group: str, version: str, generator: str, clusters: str, concepts: str, labels: str, dry_run: bool):
    create_logger_file(Path.cwd(), 'delete')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    if clusters:
        cluster_list = clusters.split(',')
        delete(api, project_id=project.id, version=version, generator=generator, group=group,
               clusters=cluster_list, dry_run=dry_run)
    if concepts:
        concept_list = concepts.split(',')
        delete(api, project_id=project.id, version=version, generator=generator, group=group,
               concepts=concept_list, dry_run=dry_run)
    if labels:
        label_list = labels.split(',')
        delete(api, project_id=project.id, version=version, generator=generator, group=group,
               labels=label_list, dry_run=dry_run)

    # if deleting everything, then delete the group
    if not clusters and not concepts and not labels:
        delete(api, project_id=project.id, version=version, generator=generator, group=group, dry_run=dry_run)


@cli.command(name="iou", help='Assign from iou from one group/generator to another')
@click.option('--group-source', help='Group name, e.g. YOLOv5-MIDWATER102')
@click.option('--group-target', help='Group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--conf', default=0.2, help='Confidence threshold for iou assignment. Defaults to 0.2.')
def iou(group_source: str, group_target: str, version: str, conf: float):
    generator = 'vars-labelbot'
    create_logger_file(Path.cwd(), 'assign')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    assign_iou(api,
               project_id=project.id,
               version=version,
               generator=generator,
               conf=conf,
               group_source=group_source,
               group_target=group_target)


@cli.command(name="classify", help='Classify concepts in localizations')
@click.option('--base-dir', default=DEFAULT_BASE_DIR, help='Base directory to save all data to.')
@click.option('--group', help='Group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--generator', help='Generator name, e.g. vars-labelbot or vars-annotation')
@click.option('--model-url', help='Url of the model to use for classification.')
def assign_classify(group: str, version: str, generator: str, model_url: str, base_dir: str):
    create_logger_file(Path.cwd(), 'classify')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    classify(api, project_id=project.id, version=version, generator=generator, group=group, model_url=model_url,
             output_path=Path(base_dir))


@cli.command(name="detect", help='Detect concepts in image urls and load into database')
@click.option('--base-url', help='Base url all images are in.')
@click.option('--group', help='Group name, e.g. VB250')
@click.option('--version', default=DEFAULT_VERSION, help=f'Dataset version to assign. Defaults to {DEFAULT_VERSION}.')
@click.option('--generator', help='Generator name, e.g. vars-labelbot or vars-annotation', default='megadetector')
@click.option('--model-url', help='Url of the model to use for classification.', default='http://fasta-fasta-1d0o3gwgv046e-143598223.us-west-2.elb.amazonaws.com/predict')
def assign_detect(group: str, version: str, generator: str, model_url: str, base_url: str):
    create_logger_file(Path.cwd(), 'detect')

    # Connect to the database api
    api = init_api()

    # Find the project
    project = find_project(api, DEFAULT_PROJECT)
    info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

    # Get the localization type
    localization_types = api.get_localization_type_list(project.id)

    # the box type is the one with the name 'Boxes'
    box_type = None
    image_type = None
    for l in localization_types:
        if l.name == 'Boxes':
            box_type = l.id
            break

    # Fail if we could not find the box type
    if box_type is None:
        info(
            f'Could not find localization type "Boxes" in project {project.name}.'
            f' Do you have a localization type "Boxes" in the project {project}?')
        return

    # Find the media type
    media_types = api.get_media_type_list(project.id)

    for m in media_types:
        if m.dtype == "image":
            image_type = m.id
            break

    if image_type and box_type:
        detect(api, image_type=image_type, box_type=box_type, project_id=project.id, version=version, generator=generator, group=group,
               model_url=model_url, base_image_url=base_url)
if __name__ == '__main__':
    try:
        load_dotenv('.env')
        start_time = time.time()
        cli()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Done. Elapsed time: {elapsed_time}')
    except Exception as e:
        err(f'Exiting. Error: {e}')
        exit(-1)
