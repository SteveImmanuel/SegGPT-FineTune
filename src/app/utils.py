import time
import os

def check_file_exists(path: str, timeout: int = 120):
    '''
    Checks if a file exists at the given path.

    Args:
        path (str): Path to the file.
        timeout (int): The number of seconds to wait for the file to appear. Default is 120 seconds.

    Raises:
        FileNotFoundError: If the file does not exist.
    '''
    check_timeout = time.time() + timeout
    while (not os.path.isfile(path)) and (time.time() < check_timeout):
        time.sleep(1)

    if not os.path.isfile(path):
        raise FileNotFoundError(f'The file {path} does not exist.')
