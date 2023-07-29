# Utility to migrate media from one server to another to the same media id
import tempfile
from pathlib import Path

import tator
from tator.util import md5sum
import time
from dotenv import load_dotenv
from bio.db.tator import find_project, init_api
from bio.logger import info, err, exception, debug, create_logger_file

DEFAULT_VERSION = 'Baseline'
DEFAULT_PROJECT = '901103-biodiversity'

def upload_and_create_media(api, project_id: int, type_id: int, media_id:int, image_path: Path, attributes: dict = None):
    try:
        info(f'Uploading {image_path.as_posix()} to project {project_id}')
        for progress, response in tator.util.upload_media(api,
                                                          section="All Media",
                                                          type_id=type_id,
                                                          media_id=media_id,
                                                          attributes=attributes,
                                                          fname=image_path.name,
                                                          path=image_path.as_posix(),
                                                          timeout=120):
            debug(f"Upload progress: {progress}%")
        info(response.message)
    except Exception as e:
        if 'object has no attribute' not in str(e):
            err(f"Error uploading {image_path.name}: {e}")
            raise e

if __name__ == '__main__':

    start_time = time.time()
    create_logger_file(Path.cwd(), 'migrate_media')

    try:
        # Connect to the database apis
        load_dotenv('.env.digits')
        api_remote = init_api()

        # Find the project
        project = find_project(api_remote, DEFAULT_PROJECT)
        info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

        # Get all the media in this project
        num_media = api_remote.get_media_count(project=project.id)
        info(f'Found {num_media} media in project {project.name}')

        media = api_remote.get_media_list(project.id)

        # Get all the unique media names
        media_names = set([m.name for m in media])
        info(f'Found {len(media_names)} unique media names')

        # Keep all media with unique names
        media_copy = []
        for m in media:
            if m.name in media_names:
                media_copy.append(m)
                media_names.remove(m.name)

        # Get the image type
        load_dotenv('.env')
        api = init_api()

        media_types = api.get_media_type_list(project.id)
        image_type = None
        for t in media_types:
            if t.dtype == "image":
                image_type = t.id
                break


        # Iterate through the media and import them to the new server which should have the same media ids
        # Work in a temporary directory
        with tempfile.TemporaryDirectory() as out_path:
            info(f'Using temporary directory: {out_path}')

            # Iterate through the media and import them to the new server which should have the same media ids
            for m in media_copy:
                info(f'Importing media id: {m.id} with name: {m.name}')

                # Download the media
                image_out_path = Path(out_path) / m.name

                for progress in tator.util.download_media(api_remote, m, image_out_path.as_posix()):
                    debug(f"{m.id} download progress: {progress}%")

                # Upload the media
                # Keep video_reference_uuid and index_elapsed_time_millis the same from the original media
                attributes = {'video_reference_uuid': m.attributes['video_reference_uuid'],
                                'index_elapsed_time_millis': m.attributes['index_elapsed_time_millis']}

                upload_and_create_media(api, project.id, image_type, m.id, image_out_path, attributes)

                image_out_path.unlink()


    except Exception as e:
        exception(e)