import logging
from time import strftime, localtime
from urllib.request import urlopen  # or maybe use urllib2

from fastai.text import *


def define_paths():
    if "kwargs" not in locals():
        kwargs = {}
    ########################### ONLY PART NEEDS SETUP ##############
    PARENT_PATH = os.environ["PARENT_PATH"]
    MODEL_FOLDER = os.path.join(PARENT_PATH, "models")

    make_sure_path_exists(MODEL_FOLDER)
    LEARNER_REMOTE_PATH = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/ulmfit_final_files/models/3_features_80pc.pkl"
    MODEL_NAME = "3_features_80pc.pkl"
    LOCAL_MODEL_FULL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

    os.environ["MODEL_FOLDER"] = MODEL_FOLDER

    LEARNER_REMOTE_PATH1 = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/ulmfit_final_files/models/3_features_80pc.pkl"
    os.environ["model1.3_features"] = LEARNER_REMOTE_PATH

    MODEL_NAME1 = "3_features_80pc.pkl"
    os.environ["MODEL_NAME1"] = MODEL_NAME

    LOCAL_MODEL_FULL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)
    os.environ["LOCAL_MODEL_FULL_PATH"] = LOCAL_MODEL_FULL_PATH
    LOG_PATH = os.path.join(os.environ["PARENT_PATH"], "logs")
    os.environ["LOG_PATH"] = LOG_PATH

    BASE_DATA_PATH = os.path.join(PARENT_PATH, "data")
    BASE_DATA_FILE = os.path.join(BASE_DATA_PATH, "products.csv")
    make_sure_path_exists(BASE_DATA_PATH)
    os.environ["BASE_DATA_PATH"] = BASE_DATA_PATH
    os.environ["BASE_DATA_FILE"] = BASE_DATA_FILE

    raw_data_url = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/data_on_public/full/products.csv"
    os.environ["fulldata"] = raw_data_url

    ##############################################################
    logger = define_logger(log_dir=LOG_PATH)
    #################### MODEL GOES ALWAYS TOGETHER WITH SPECIFIC LEARNER AND DATASET

    model_feature_names = os.environ["model_feature_names"].split(",")
    logger.info(f"Model 1 is defined as {MODEL_NAME1}")
    logger.info(f"Model 1 feature names are {model_feature_names}")
    logger.info(f"Model 1 remote path is defined as {LEARNER_REMOTE_PATH1}")

    logger.info(f"Model folder is defined as {MODEL_FOLDER}")
    logger.info(f"Raw data remote path is defined as {raw_data_url}")

    return logger


def download_from_url(logger, local_path, remote_path, mode="data", **kwargs):
    logger.info(f"Starting to download {mode} from {remote_path} remote resource")
    if mode == "data":
        FILENAME = "products.csv"
        folder_to_check = os.environ["BASE_DATA_PATH"]

    elif mode == "model":
        FILENAME = os.environ["MODEL_NAME1"]
        folder_to_check = os.environ["MODEL_FOLDER"]

    if FILENAME not in os.listdir(folder_to_check):
        logger.debug(f"It seems the {mode} hasn't been downloaded yet, so we will do so now: \n")
        logger.debug(f"=================DOWNLOADING THE {mode} ==================")
        filedata = urlopen(remote_path)
        datatowrite = filedata.read()
        with open(local_path, 'wb') as f:
            f.write(datatowrite)

        if FILENAME not in os.listdir(folder_to_check):
            logger.warn(f"It seems {mode} download didn't work, please check the function!")

    else:
        logger.info(f"{mode} is present in {folder_to_check}, proceed normally!")


def define_logger(log_dir, log_to_this_file="l", log_to_file=True):
    make_sure_path_exists(log_dir)
    running_mode = os.environ["running_mode"]
    log_to_this_file = f"{log_to_this_file}_{running_mode}"
    log_to_this_file = os.path.join(log_dir, log_to_this_file)
    log_to_this_file = log_to_this_file + strftime("%Y_%m_%d_%H_%M", localtime()) + ".log"
    logger = logging.getLogger('l')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if log_to_file == True:
        fh = logging.FileHandler(log_to_this_file)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')  # %(asctime)s
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        print(f"Logging all updates to {log_to_this_file} this file")
        os.environ["logger_path"] = log_to_this_file
    return logger


def make_sure_path_exists(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            print(f"{dir} path didn't exist, created! Proceed now!")
        except OSError as exception:
            print("Directory Creation Exception")


def load_model_from_local_path(logger):
    model_name = os.environ["MODEL_NAME1"]
    model = load_learner(path=os.environ["MODEL_FOLDER"],
                         file=model_name, **{"num_workers": 0})  # OR NUM WORKERS
    if "model" in locals():
        logger.info(f"Model {model_name} loaded!")
    else:
        logger.error("Something went wrong loading the model")
    return model


def download_and_load_model(logger):
    download_from_url(logger=logger, mode="model", local_path=os.environ["LOCAL_MODEL_FULL_PATH"], \
                      remote_path=os.environ["model1.3_features"])
    model = load_model_from_local_path(logger)
    logger.info("Model Loaded!")
    return model
