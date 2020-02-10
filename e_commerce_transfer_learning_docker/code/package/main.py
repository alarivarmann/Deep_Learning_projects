import env_setup as e
from predict import *
from preprocess import *


def make_sure_path_exists(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            print(f"{dir} path didn't exist, created! Proceed now!")
        except OSError as exception:
            print("Directory Creation Exception")


def main(logger, target_name, skip="", amountlines=int(-1), save_splits_again=False):
    """This is the caller function of the data pipeline, producing predictions of E-Commerce Categories from a defined set of features."""
    step1 = step2 = {}
    # data_object is the dictionary containing outputs of data pipeline steps
    data_object = {"step1": step1, "step2": step2}

    skip_steps = skip.split(' ')

    logger.info(" ----------- STEP 1 : DATA LOADING AND PROCESSING  --------------")

    if "data_read" not in skip_steps:
        # model1.3_features
        raw_data_local_path = os.environ["BASE_DATA_FILE"]
        e.download_from_url(logger, mode="data", local_path=raw_data_local_path,
                            remote_path=os.environ["fulldata"])
        print(f"THE RAW DATA LOCAL PATH IS {raw_data_local_path}")
        if amountlines != -1:
            rows_to_keep = list(range(amountlines))
            df = pd.read_csv(raw_data_local_path, skiprows=lambda x: x not in rows_to_keep)
            logger.info(
                f"----- READING {rows_to_keep} lines of  DATA FROM  path {raw_data_local_path} COMPLETED ------------------ ")
        else:
            df = pd.read_csv(raw_data_local_path)
            logger.info(f"----- READING FULL DATA FROM  path {raw_data_local_path} COMPLETED ------------------ ")

    cols_to_drop = ['ean', 'url', 'price', './._prods.csv']
    logger.info(f"Going to drop these columns if they exist: {cols_to_drop}\n")
    df = drop_if_exists(df, cols=cols_to_drop)
    initial_feature_names = [x for x in df.columns.tolist() if x != target_name]
    logger.info("Initially, the features in the dataframe are defined as follows")
    logger.info(str(initial_feature_names))
    if df is None:
        logger.error("THERE IS NO DATA TO BE READ, PROGRAM STOPPING!")
        return

    tqdm.pandas(desc='PROGRESS>>>')
    data_1 = step_1_data_processing(df, logger=logger)

    step1["data"] = data_1
    data_object["data_read"] = step1
    logger.info(str(data_1.head()))
    logger.info(" ----------- STEP 1 COMPLETE : DATA LOADING AND PROCESSING  --------------")

    logger.info(" ----------- STEP 2 BEGINS : DATA PRE-PROCESSING  --------------")

    """ Preferred mode : read data from disk"""
    if "data_process" not in skip_steps:
        data_2 = step_2_save_splits_from_dataframe(save_splits_again=save_splits_again, \
                                                   logger=logger, df=data_1)
    else:
        data_2 = []
    step2["data"] = data_2
    data_object["data_process"] = step2

    logger.info(f" ----------- STEP 2, preprocessing, ENDED.  --------------")

    logger.info(
        " ----------- STEP 3 : DOWNLOADING AND LOADING PRETRAINED HIGH DIMENSIONAL CLASSIFICATION MODEL  --------------")
    if "model" not in skip_steps:
        model = e.download_and_load_model(logger=logger)
        logger.info("------- MODEL HAS BEEN DOWNLOADED AND LOADED, PROCEED WITH MACHINE LEARNING ----------")
    data_object["model"] = model

    if "predictions" not in skip_steps:
        DATA_to_validate = data_2["val"]
        FEATURES_to_validate, labels = select_globally_defined_features_and_target(data=DATA_to_validate)

        obtain_predictions_in_loop(logger, model, features=FEATURES_to_validate, labels=labels)
        data_object["features"] = FEATURES_to_validate
        data_object["labels"] = labels
        # data_object["predictions"] = predictions # save the predictions here somewhere
    return data_object


if __name__ == "__main__":
    PARENT_PATH = os.getcwd()
    make_sure_path_exists(PARENT_PATH)
    os.environ["PARENT_PATH"] = PARENT_PATH
    os.chdir(PARENT_PATH)
    print(f"Switched working directory to {PARENT_PATH}, STARTING PROGRAM")

    ################################ RUNTIME PARAMETERS ##########################

    model_feature_names = "brand,provider"
    running_mode = "production"
    os.environ["running_mode"] = running_mode
    if running_mode == "production":
        model_feature_names = f"combined_name_description,{model_feature_names}"
    os.environ["model_feature_names"] = model_feature_names
    logger = e.define_paths()
    #####################################################################################
    introduction = "\n WELCOME TO THE e-Commerce Data Taxonomy Tree " \
                   "Advanced AI-based (Re-)Mapper \n \n by \n \n Alari Varmann // Integrify \n"
    print(introduction)
    logger.info(introduction)
    logger.info(
        f"Runtime parameters have been defined: running mode {running_mode}, feature names {model_feature_names}")
    logger.info(f"Starting the Main Program of the Intelligent E-Commerce Data Re-Mapper!")
    data_pipeline_object = main(logger=logger, target_name="mapped_id", save_splits_again=True)

    ###################################################33
    outroduction = "\n Hopefully these mappings were useful. " \
                   "Give your feedback about them to alari.varmann@integrify.io \n"
    print(outroduction)
    logger.info(outroduction)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-skip","--skip-steps", default="2", type=str,help="Steps to skip separated by space")
    # args = parser.parse_args()
