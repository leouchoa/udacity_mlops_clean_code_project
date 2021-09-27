"""
This module holds all the tests of the functions
created in the `churn_library.py` file. To run it
use `python churn_script_logging_and_tests.py`

Author: Leonardo Pedreira
Date: Septembre 2021
"""
import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        churn_df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert churn_df.shape[0] > 0
        assert churn_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        churn_df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(churn_df)
        assert os.path.isfile('images/eda/churn_distribution.png')
        assert os.path.isfile('images/eda/customer_age_distribution.png')
        assert os.path.isfile('images/eda/marital_status_distribution.png')
        assert os.path.isfile('images/eda/total_transaction_distribution.png')
        assert os.path.isfile('images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    cat_list = ['Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
                ]
    try:
        churn_df = cls.import_data('data/bank_data.csv')
        churn_df = cls.encoder_helper(churn_df, cat_list, response='Churn')
        assert set(name + '_Churn' for name in cat_list
                   ).issubset(churn_df.columns)
        logging.info('Testing encoder_helper: SUCCESS')
    except KeyError as err:
        logging.error(
            'Testing encoder_helper: There is not some categorical column')
        raise err
    try:
        assert churn_df[[name + '_Churn' for name in cat_list]].shape[0] > 0
        assert churn_df[[name + '_Churn' for name in cat_list]].shape[1] > 0
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: The dataframe does not appear to have rows and columns')
        raise err
    try:
        churn_df_new = cls.encoder_helper(churn_df, cat_list, response = 'Churn')
        assert sum(churn_df_new.columns.str.contains('_Churn')) == 5
    except AssertionError as err:
        logging.error("Testing encoder_helper: The response not work")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    try:
        churn_df = cls.import_data('./data/bank_data.csv')

        cat_list = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category'
                    ]
        churn_df = cls.encoder_helper(churn_df, cat_list, response='Churn')
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            churn_df, response='Churn')
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_train, pd.Series)
        logging.info(
            'Testing perform_feature_engineering: SUCCESS type feature')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: There is problem with get '
            'some feature')
        raise err
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] == 19
        assert X_test.shape[0] > 0
        assert X_test.shape[1] == 19
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            'Testing perform_feature_engineering: SUCCESS shape feature')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: The dataframe does not '
            'appear to have the right rows and columns')
        raise err


def test_train_models():
    '''
    test train_models
    '''

    try:
        churn_df = cls.import_data('./data/bank_data.csv')

        cat_list = ['Gender',
                    'Education_Level',
                    'Marital_Status',
                    'Income_Category',
                    'Card_Category'
                    ]
        churn_df = cls.encoder_helper(churn_df, cat_list, response='Churn')
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            churn_df, response='Churn')
        cls.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('models/rfc_model.pkl')
        assert os.path.isfile('models/logistic_model.pkl')
        assert os.path.isfile('model_eval_metrics/rf_metrics.csv')
        assert os.path.isfile('model_eval_metrics/lr_metrics.csv')
        assert os.path.isfile('images/results/feature_importances.png')
        assert os.path.isfile('images/results/roc_curve_result.png')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: Some file of train models not found in the")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
