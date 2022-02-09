'''
Script for testing churn_library

Author: Diego Alfaro
Date: 02/2022
'''


import os
import logging
from churn_library import ChurnPredictor

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(churn_predictor):
    '''
    test churn_predictor instance creation - this example is completed
    for you to assist with the other test functions

    input:
        ChurnPredictor class
    output:
        ChurnPredictor instance
    '''
    try:
        churn_mdl = churn_predictor("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS\n")
        return churn_mdl
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found\n")
        raise err
    except NameError as err:
        logging.error(
            "Testing import_data: constants.py must have lists cat_columns and quant_columns\n")
        raise err

    try:
        assert churn_mdl.churn_df.shape[0] > 0
        assert churn_mdl.churn_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns\n")
        raise err


def test_eda(churn_mdl):
    '''
    test on eda module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
    output:
        None
    '''
    try:
        churn_mdl.general_eda()
        assert os.path.isfile("./images/eda/corr.png")
        logging.info("Testing general_eda: SUCCESS\n")
    except FileNotFoundError:
        logging.error(
            "Testing general_eda: Missing directory ./images/eda to store correlation heatmap.\n")
    except AssertionError:
        logging.error(
            "Testing general_eda: Missing output file ./images/eda/corr.png\n")


def test_feature_dist(churn_mdl, feature):
    '''
    test on feature_dist module of ChurnPredictor instance
    input:
        churn_mdl: churn_predictor instance
        feature: input for feature_dist module
    output:
        None
    '''
    try:
        assert isinstance(feature, str)
        churn_mdl.feature_dist(feature)
        logging.info("Testing feature_dist: SUCCESS\n")
    except KeyError:
        logging.error(
            f"Testing feature_dist: {feature} not in data column names.\n")
    except AssertionError:
        logging.info(f"Testing feature_dist: input is {feature}")
        logging.error("Testing feature_dist: input must be string.\n")
    except FileNotFoundError:
        logging.error(
            "Testing feature_dist: Missing directory ./images/eda to store histogram.\n")


def test_bivariate_dist(churn_mdl, in_test):
    '''
    test on bivariate_dist module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
        in_test: input for bivariate_dist module
    output:
        None
    '''
    try:
        aux = iter(in_test)
        assert len(in_test) == 2
        churn_mdl.bivariate_dist(in_test)
        logging.info("Testing bivariate_dist: SUCCESS\n")
    except KeyError:
        logging.error(
            f"Testing bivariate_dist: input {in_test} not in data column names.\n")
    except TypeError:
        logging.info(f"Testing bivariate_dist: input is {in_test}")
        logging.error("Testing bivariate_dist: input must be iterable.\n")
    except AssertionError:
        logging.info(
            f"Testing bivariate_dist: input is of length {len(in_test)}")
        logging.error(
            "Testing bivariate_dist: input iterable must have 2 entries.\n")
    except FileNotFoundError:
        logging.error(
            "Testing bivariate_dist: Missing directory ./images/eda to store scatter plot.\n")


def test_encoder_helper(churn_mdl, in_test):
    '''
    test encoder_helper module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
        in_test: input for encoder_helper module
    output:
        churn_mdl: ChurnPredictor instance
    '''
    try:
        assert isinstance(in_test, str)
        assert in_test not in churn_mdl.churn_df.columns
        churn_mdl.encoder_helper(in_test)
        logging.info("Testing ecoder_helper: SUCCESS\n")
        return churn_mdl
    except AssertionError:
        logging.info(f"Testing encoder_helper: input is {in_test}")
        logging.error(
            "Testing encoder_helper: input must be string, not in column names.\n")
        return None


def test_feature_engineering(churn_mdl, in_test):
    '''
    test feature_engineering module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
        in_test: input for feature_engineering module
    output:
        churn_mdl: ChurnPredictor instance
    '''
    try:
        assert isinstance(in_test, float)
        churn_mdl.feature_engineering(in_test)
        logging.info("Testing feature_engineering: SUCCESS\n")
        return churn_mdl
    except AssertionError:
        logging.info(f"Testing feature_engineering: input is {in_test}")
        logging.error(
            "Testing feature_engineering: input must be float or left empty.\n")
        return None
    except ValueError:
        logging.error(
            "Testing feature_engineering: must call encoder_helper module.\n")
        return None


def test_train_models(churn_mdl,
                      n_estimators=(200, 300),
                      max_depth=(4, 5),
                      max_iter=1000
                      ):
    '''
    test train_models module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
        n_estimators: iterable with values for gridsearch on RF
        max_depth: iterable with values for gridsearch on RF
        max_iter: integer with # of iterations for logit training.
    output:
        churn_mdl: ChurnPredictor instance
    '''
    try:
        assert churn_mdl.x_train is not None
        churn_mdl.train_models(n_estimators=n_estimators,
                               max_depth=max_depth,
                               max_iter=max_iter
                               )
        logging.info("Testing train_models: SUCCESS\n")
        return churn_mdl
    except AssertionError:
        logging.error(
            "Testing train_models: must call feature_engineering module.\n")
        return None
    except ValueError:
        logging.error(
            "Testing train_models: inputs must be iterables containing int.\n")
        return None


def test_classification_report_image(churn_mdl):
    '''
    test classification_report_image module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
    output:
        None
    '''
    try:
        assert churn_mdl.rfc is not None
        churn_mdl.classification_report_image()
        logging.info("Testing classification_report_image: SUCCESS\n")
    except AssertionError:
        logging.error(
            "Testing classification_report_image: must call train_models module.\n")
    except FileNotFoundError:
        logging.error(
            "Testing classification_report_image: Missing directory ./images/results\n")


def feature_importance_plot(churn_mdl):
    '''
    test feature_importance_plot module of ChurnPredictor instance
    input:
        churn_mdl: ChurnPredictor instance
    output:
        None
    '''
    try:
        assert churn_mdl.rfc is not None
        churn_mdl.feature_importance_plot()
        logging.info("Testing feature_importance_plot: SUCCESS\n")
    except AssertionError:
        logging.error(
            "Testing feature_importance_plot: Must call train_models module.\n")
    except FileNotFoundError:
        logging.error(
            "Testing feature_importance_plot: Missing directory ./images/results\n")


if __name__ == "__main__":
    #                                                       Expected behavior#
    #######################
    CHURN_MDL = test_import(ChurnPredictor)  # Success

    test_eda(CHURN_MDL)  # Success

    test_feature_dist(CHURN_MDL, "Card_Category")  # Success: cat var
    test_feature_dist(CHURN_MDL, "hello")  # Error: not in columns
    test_feature_dist(CHURN_MDL, "Dependent_count")  # Success: quant var
    test_feature_dist(CHURN_MDL, ["Dependent_count"])  # Error: incorrect input

    test_bivariate_dist(CHURN_MDL, ["Customer_Age",
                                    "Months_on_book"])  # Success
    test_bivariate_dist(CHURN_MDL, ["Customer",  # Error: not in columns
                                    "Months"])
    # Error: incorrect length
    test_bivariate_dist(CHURN_MDL, ["Months_on_book"])
    # Error: incorrect input type
    test_bivariate_dist(CHURN_MDL, "Months_on_book")

    # Error: must run encoder_helper first
    test_feature_engineering(CHURN_MDL, .1)
    test_encoder_helper(CHURN_MDL, [1])  # Error: incorrect input type
    CHURN_MDL = test_encoder_helper(CHURN_MDL, "churn")  # Success

    test_train_models(CHURN_MDL)  # Error: must run feature_engineering first

    test_feature_engineering(CHURN_MDL, ".1")  # Error: incorrect input type
    CHURN_MDL = test_feature_engineering(CHURN_MDL, .1)  # Success

    # Error: must run train_models first
    test_classification_report_image(CHURN_MDL)

    test_train_models(CHURN_MDL, max_depth=5)  # Error: incorrect input type
    CHURN_MDL = test_train_models(CHURN_MDL)  # Success

    test_classification_report_image(CHURN_MDL)  # success
    feature_importance_plot(CHURN_MDL)  # success
