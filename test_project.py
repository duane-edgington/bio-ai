import time
import tator
from pathlib import Path

from bio.db import init_api
from bio.logger import create_logger_file, info


def main():
    create_logger_file(Path.cwd(), 'test_project')
    print("Hello World!")

    # Code below will print out all projects and their metadata in the database
    # Connect to the database api
    api = init_api()
    #
    # # List all projects
    projects = api.get_project_list()
    info(projects)
    for p in projects:
        print(p)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env')
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done. Elapsed time: {elapsed_time}')
