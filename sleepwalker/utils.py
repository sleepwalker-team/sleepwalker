from datetime import datetime
import importlib
import logging
import os
import inspect

# We use a singleton-pattern logger in this project, because it is unlikely that we need multiple loggers (ML-related logs such as loss curves are handled differently anyways). We expect people to use the logger as:
#   from utils import logger
#   logger.info("...")
#   logger.debug("...") 
# etc.
def get_logger(path_to_logfile="./sleepwalker.log"):
    """
    Create and configure a logger object.

    Args:
        folder (str, optional): The folder path where the log file will be saved. Defaults to ".".
        name (str, optional): The name of the log file. Defaults to "sleepwalker".

    Returns:
        logging.Logger: The configured logger object.
    """

    # Create custom logger logging all five levels 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Define format for logs
    # NOTE: If (for some reason) we change fmt, we have to change log_prefix as well. They are currently hard-coded to match
    fmt = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)2d | %(message)s'

    # Create stdout handler for logging to the console (logs all five levels)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(logging.Formatter(fmt))

    # Create file handler for logging to a file (logs all five levels)
    file_handler = logging.FileHandler(path_to_logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))

    # Add both handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    # Function to update the log file path
    def update_log_file_path(new_path):
        file_handler.close()  
        file_handler.baseFilename = new_path  

    # Function to generate a consistent log prefix for tqdm
    # NOTE: If (for some reason) we change fmt, we have to change this function as well. They are currently hard-coded to match
    def log_prefix(level="INFO"):
        # Retrieve caller frame
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        prefix = f"{now} | {level:<4} | {filename}:{lineno:2d} | "
        return prefix

    # Attach the update function to the logger object
    logger.update_log_file_path = update_log_file_path
    logger.log_prefix = log_prefix

    return logger

# The singleton logger 
logger = get_logger()

def load_class(module_path, name):
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as e:
        raise ImportError(f"Cannot import '{name}' from module '{module_path}': {e}")