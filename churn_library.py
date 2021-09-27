'''
Goal: create a model to predict churn, while saving
every result found in the pipeline.

Author: Leonardo Uchôa Pedreira
Date: 12/Sept/2021
'''


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import os
os.environ['QT_QPA_PLATFORM']='offscreen'

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(path_to_csv: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            churn_df: pandas dataframe
    '''

    churn_df = pd.read_csv(path_to_csv)

    churn_df['Churn'] = churn_df['Attrition_Flag'].apply(
        lambda x: 0 if x == 'Existing Customer' else 1)

    return churn_df


def perform_eda(churn_df: pd.DataFrame) -> None:
    '''
    perform eda on churn_df and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    '''

    # plot+save churn vs no churn count
    fig = plt.figure(figsize=(20, 10))
    churn_df['Churn'].hist()
    fig.savefig(
        'images/eda/churn_distribution.png',
        bbox_inches='tight',
        dpi=150)
    plt.clf()

    # plot+save costumer age histogram
    fig = plt.figure(figsize=(20, 10))
    churn_df['Customer_Age'].hist()
    fig.savefig(
        'images/eda/customer_age_distribution.png',
        bbox_inches='tight',
        dpi=150)
    plt.clf()

    # plot+save total transaction count
    fig = plt.figure(figsize=(20, 10))
    sns.distplot(churn_df['Total_Trans_Ct'])
    fig.savefig(
        'images/eda/total_transaction_distribution.png',
        bbox_inches='tight',
        dpi=150)
    plt.clf()

    # plot+save marital status
    sns.barplot(
        x=churn_df['Marital_Status'].value_counts().index,
        y=churn_df['Marital_Status'].value_counts('normalize').values,
        data=churn_df
    )
    fig.savefig(
        'images/eda/marital_status_distribution.png')
    plt.clf()

    # plot+save heatmap
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(churn_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.tight_layout()
    fig.savefig('images/eda/heatmap.png')
    plt.clf()


def encoder_helper(churn_df: pd.DataFrame, category_lst: list, response: str):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could ...
            ... be used for naming variables or index y column]

    output:
            churn_df: pandas dataframe with new columns for
    '''
    for categorical in category_lst:
        cat_groups = churn_df.groupby(categorical).mean()[response]
        cat_values = [cat_groups.loc[val] for val in churn_df[categorical]]
        new_categorical = '_'.join([categorical, response])
        churn_df[new_categorical] = cat_values

    return churn_df


def perform_feature_engineering(
        churn_df: pd.DataFrame,
        response: str) -> tuple:
    '''
    input:
              churn_df: pandas dataframe
              response: string of response name [optional argument that could ...
              ... be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = churn_df[response]
    X = pd.DataFrame()

    X[keep_cols] = churn_df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest Results
    # fig = plt.rc('figure', figsize=(5, 5))
    # plt.text(
    #     0.01,
    #     1.25,
    #     str('Random Forest Train'),
    #     {'fontsize': 10},
    #     fontproperties = 'monospace'
    #     )
    # plt.text(
    #     0.01,
    #     0.05, 
    #     str(classification_report(y_train, y_train_preds_rf)), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     ) # approach improved by OP -> monospace!
    # plt.text(
    #     0.01, 
    #     0.6, 
    #     str('Random Forest Test'), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     )
    # plt.text(
    #     0.01, 
    #     0.7, 
    #     str(classification_report(y_test, y_test_preds_rf)), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     ) # approach improved by OP -> monospace!
    # plt.axis('off')
    # plt.savefig('images/results/rf_results.png')
    # plt.clf()

    # # Logistic Regression Results
    # plt.rc('figure', figsize=(5, 5))
    # plt.text(
    #     0.01,
    #     1.25,
    #     str('Logistic Regression Train'),
    #     {'fontsize': 10},
    #     fontproperties = 'monospace'
    #     )
    # plt.text(
    #     0.01,
    #     0.05, 
    #     str(classification_report(y_train, y_train_preds_lr)), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     ) # approach improved by OP -> monospace!
    # plt.text(
    #     0.01, 
    #     0.6, 
    #     str('Logistic Regression Test'), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     )
    # plt.text(
    #     0.01, 
    #     0.7, 
    #     str(classification_report(y_test, y_test_preds_lr)), 
    #     {'fontsize': 10}, 
    #     fontproperties = 'monospace'
    #     ) # approach improved by OP -> monospace!
    # plt.axis('off')
    # plt.savefig('images/results/lr_results.png')
    # plt.clf()

    report_metric_name_col = [
        'class_0',
        'class_1',
        'accuracy',
        'macro avg',
        'weighted avg'
    ]

    rf_report = classification_report(
        y_test, 
        y_test_preds_rf, 
        output_dict=True
        )
    rf_report = pd.DataFrame(rf_report).transpose()
    rf_report['metric_name'] = report_metric_name_col
    rf_report.to_csv('model_eval_metrics/rf_metrics.csv', index = False)

    lr_report = classification_report(
        y_test, 
        y_test_preds_lr, 
        output_dict=True
        )
    lr_report = pd.DataFrame(lr_report).transpose()
    lr_report['metric_name'] = report_metric_name_col
    lr_report.to_csv('model_eval_metrics/lr_metrics.csv', index = False)



def feature_importance_plot(model, X_data: pd.DataFrame) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create Plot
    plt.figure(figsize=(20, 15))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig('images/results/feature_importances.png')
    plt.clf()

    # shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('images/results/feature_importance_shap.png')
    plt.clf()


def plot_roc_curve_and_save(rfc_model,
                            lrc_model,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> None:
    """
    Use plot_roc_curve to create and save the roc curves for logistic regression
    and random forest model
    :param rfc_model: random forest model
    :param lrc_model: logistic regression model
    :param X_test: X testing data
    :param y_test: y testing data
    :return: None
    """
    plt.figure(figsize=(15, 8))
    ax_plot_arg = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax_plot_arg, alpha=0.8)
    plot_roc_curve(lrc_model, X_test, y_test, ax=ax_plot_arg)
    plt.savefig('images/results/roc_curve_result.png')
    plt.clf()


def train_models(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=10000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, 
        param_grid=param_grid, 
        # cv=5
        cv=2
        )
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    # lr_model = joblib.load('./models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(rfc_model, X_test)

    plot_roc_curve_and_save(rfc_model = rfc_model,
                            lrc_model = lrc,
                            X_test = X_test,
                            y_test= y_test)


if __name__ == "__main__":

    cat_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    df = import_data('data/bank_data.csv')
    df = encoder_helper(df, cat_list, response="Churn")

    perform_eda(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response="Churn")

    train_models(X_train, X_test, y_train, y_test)
