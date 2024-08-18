import logging
import os

def setup_logger(log_dir="logs", log_filename="app.log"):
    """
    Set up a logger that logs messages to both a file and the console.
    
    Parameters:
    - log_dir: Directory where the log file will be stored.
    - log_filename: Name of the log file.
    
    Returns:
    - logger: Configured logger instance.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path for the log file
    log_path = os.path.join(log_dir, log_filename)
    
    # Create a custom logger
    logger = logging.getLogger("AppLogger")
    
    # Set the log level
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    
    # Set the log level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
