from pathlib import Path
from bio.logger import create_logger_file


def main():
    create_logger_file(Path.cwd(), 'test_project')
    print("Hello World!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env')
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done. Elapsed time: {elapsed_time}')
