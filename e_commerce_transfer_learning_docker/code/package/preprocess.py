import itertools
import os
import pickle
import re
from time import strftime, localtime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def apply_fun(dict_: dict, fun, **kwargs):
    transformed_dict = dict((k, fun(v, **kwargs)) for k, v in dict_.items())
    return transformed_dict


def replace_global_missing_features_with_empty(df):
    cols = os.environ["model_feature_names"].split(",")
    df.loc[:, cols] = df.loc[:, cols].replace(np.nan, '', regex=True)
    return df


def drop_if_exists(df, cols):
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()


def savePickle(data, picklefile):
    with open(picklefile, "wb") as f:
        pickle.dump(data, f)
    f.close()


def step_1_data_processing(df, logger):
    model_feature_names = os.environ["model_feature_names"].split(",")
    if "logger" in globals():
        print("logger is globally defined!")
    elif "logger" in locals():
        print("logger is not globally defined, but it is locally")
    else:
        print("logger is not defined at all!")
    featurestring = f"Retaining only {model_feature_names} out of all features. Also replacing NAN with empty."
    logger.info(featurestring)
    running_mode = os.environ["running_mode"]
    if running_mode == "production":
        df["cleaned_name"] = df.progress_apply(lambda row: text_cleaner(str(row["name"]).lower()), axis=1)
        df["cleaned_description"] = df.progress_apply(lambda row: text_cleaner(str(row["description"]).lower()), axis=1)
        df["combined_name_description"] = df.progress_apply(
            lambda row: str(row["cleaned_name"]).lower() + str(row["cleaned_description"]).lower(), axis=1)

    print(f"Model 1 feature names are {model_feature_names}")

    df = replace_global_missing_features_with_empty(df)
    target_with_feature_names = [i for i in itertools.chain(model_feature_names, ["mapped_id"])]
    df = df.loc[:, target_with_feature_names]
    logger.info("Category preprocessing function finished!")
    return df


def step_2_save_splits_from_dataframe(df, save_splits_again, logger, \
                                      target_name=["mapped_id"]):
    model_feature_names = os.environ["model_feature_names"].split(",")
    df.loc[:, target_name] = df.loc[:, target_name].astype(int)
    df = df.groupby(target_name).filter(lambda x: len(x) > 1)
    logger.info(
        f"Target classes with only 1 count have been filtered out. Resulting dataframe has {df.shape[0]} elements")
    features = df.loc[:, model_feature_names].reset_index(drop=True)
    target = df.loc[:, target_name].reset_index(drop=True)
    time_now = strftime("%Y_%m_%d_%H_%M", localtime())
    logger.info(f"Time now is {time_now}")
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.20,
                                                      random_state=42, stratify=target)

    ml_splits = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}

    # Here we are going to create the output dictionary to save the machine learning splits in one Pickle file
    train_val_dict = {}
    train_val_dict["train"] = pd.concat([ml_splits.get("X_train"), ml_splits.get("y_train")], axis=1)
    train_val_dict["val"] = pd.concat([ml_splits.get("X_val"), ml_splits.get("y_val")], axis=1)
    logger.info("Header of the train-validation machine learning splits dictionary look like this: \n")
    logger.info(str(apply_fun(train_val_dict, fun=pd.DataFrame.head)))

    if save_splits_again == True:
        time_now = strftime("%Y_%m_%d_%H_%M", localtime())
        splits_end = f"ml_splits_{time_now}.pkl"
        base_data_path = os.environ["BASE_DATA_PATH"]
        splits_path = os.path.join(base_data_path, splits_end)
        savePickle(data=train_val_dict, picklefile=splits_path)
        logger.info(f"Saving ML Splits to the path {splits_path}")
    return train_val_dict
