import logging
import os
import sys

def get_logger(module_name):
   
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    log_dir = os.path.join(root_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(module_name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | [%(name)s] | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_file_path = os.path.join(log_dir, 'project.log')
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
