import os

from tqdm import tqdm


def select_globally_defined_features_and_target(data, target_name="mapped_id"):
    labels = data.loc[:, target_name]
    feature_names = os.environ["model_feature_names"].split(",")
    features = data.loc[:, feature_names].reset_index(drop=True)
    return features, labels


def get_one_prediction(model, feature_row):
    one_prediction = model.predict(feature_row)[0].obj
    # print(f"Model Prediction value is: {one_prediction}")
    return one_prediction


def obtain_predictions_in_loop(logger, model, features, labels):
    incorrects = 0
    featurecount = features.shape[1]
    for i, feature_row in tqdm(features.iterrows(), total=features.shape[0]):
        prediction = get_one_prediction(model, feature_row)
        logger.info(f"Feature row {i} looks like this: \n")
        logger.info(f"{str(feature_row.values)}\n")
        logger.info(f"The corresponding prediction row {i} is: {prediction}\n")
        label_i = labels.iloc[i]
        if prediction == label_i:
            answer = "correct"
            msg = f"The predicted tree path {prediction}is {answer}"
        else:
            incorrects += 1
            incorrects_ratio = incorrects, i
            answer = "INCORRECT"
            msg = f"{answer} tree path!"
            correctprint = f"The actual target value is: {label_i}, \n " \
                f"running incorrects {incorrects_ratio} using only {featurecount} features!"
            print(correctprint)
            logger.info(correctprint)

        # print(msg)
        logger.info(msg)
        if i == 200:
            break
