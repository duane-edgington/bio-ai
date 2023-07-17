import time

from dotenv import load_dotenv
from pathlib import Path

from bio.db.tator import init_api, find_project
from bio.logger import create_logger_file, info, err

# Default values
# The base directory is the same directory as this file
DEFAULT_BASE_DIR = Path(__file__).parent.as_posix()

DEFAULT_VERSION = 'Baseline'
DEFAULT_PROJECT = '901103-biodiversity'

if __name__ == '__main__':
    try:
        create_logger_file(Path.cwd(), 'merge_classify')
        load_dotenv('.env')
        start_time = time.time()

        group="VARSi2MAP250-06-27-2023"
        # Connect to the database api
        api = init_api()

        # Find the project
        project = find_project(api, DEFAULT_PROJECT)
        info(f'Found project id: {project.name} for project {DEFAULT_PROJECT}')

        # Get all the media in the project
        all_media = api.get_media_list(project.id)

        num_updated = 0
        for media in all_media:
            media_localizations = api.get_localization_list_by_id(project=project.id,
                                                                  localization_id_query={'media_ids': [media.id]} )

            # If any of the localizations are from the group "VARSi2MAP250-06-27-2023" then set the
            # MERGE groups to "MERGE_CLASSIFY_ONLY"
            flag_merge = False
            for localization in media_localizations:
                if localization.attributes['group'] == group:
                    flag_merge = True
                    continue
                if flag_merge and localization.attributes['group'] == 'MERGE':
                    localization.attributes['group'] = 'MERGE_CLASSIFY_ONLY'
                    localization.version = None
                    api.update_localization(localization.id, localization)
                    info(f'Updated localization {localization.id} in media {media.id} to MERGE_CLASSIFY_ONLY')
                    num_updated += 1

        info(f'Updated {num_updated} localizations to MERGE_CLASSIFY_ONLY')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Done. Elapsed time: {elapsed_time}')
    except Exception as e:
        err(f'Exiting. Error: {e}')
        exit(-1)
