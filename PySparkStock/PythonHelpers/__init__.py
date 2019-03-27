import logging
import numpy as np
import os
from datetime import datetime
import requests
import time
import re
import shutil
import threading
import math


def set_logging(level=10,
                path=None):
    format = '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if path:
        logging.basicConfig(level=level, format=format, filename=path)
    else:
        logging.basicConfig(level=level, format=format)


def add_end_backslash(folder_path):
    """

    :param folder_path:
    :return:

    add_end_backslash('/tmp')
    """
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    return folder_path


def create_folder(folder_path=None):
    if os.path.isdir(folder_path) is False:
        os.makedirs(folder_path)
    
    logging.debug(folder_path + " was created")
    
    return add_end_backslash(folder_path)



def check_all_thread_success(job_list):
    for job_i in job_list:
        
        job_i.join()
        
        if 'exitcode' in dir(job_i) and job_i.exitcode == 1:
            raise Exception(str(job_i) + ' failed')


def remove_path(path):
    """
    path = '/tmp/analysis.csv'
    :param path:
    :return:
    """
    try:
        shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)


def start_multi_threading(thread_list,
                          max_threads=20,
                          all_threads_have_to_be_success=True):
    if len(thread_list) == 0:
        return None
    
    for count_i, thread in enumerate(thread_list):
        thread.daemon = True
        thread.start()
        logging.info("""Thread {thread_id} started""".format(thread_id=thread.name))
        
        while len([thread_i for thread_i in thread_list if thread_i.is_alive()]) >= max_threads:
            time.sleep(0.1)
            logging.debug('You have run too many threads! Have a rest!!')
    
    for thread in thread_list:
        
        thread.join()
        
        if all_threads_have_to_be_success and \
                'exitcode' in dir(thread) and \
                thread.exitcode == 1:
            
            raise threading.ThreadError(thread.name + ' failed')
        else:
            logging.info(thread.name + " Is Done")