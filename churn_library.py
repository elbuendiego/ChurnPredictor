'''
Library of functions used in churn_notebook.ipynb and churn_script_logging_and_tests_solution.py

Author: Diego Alfaro
Date: 02/2022
'''

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from constants import *

sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'

class ChurnPredictor():
    '''
    ChurnPredictor class for churn modeling and analysis
    '''

    def __init__(self, pth):
        '''
        ChurnPredictor initialization requiring pth to csv data

        input:
            pth: a path to the csv
        output:
            ChurnPredictor instance
        '''
        churn_df = pd.read_csv(pth, index_col=False)
        self.churn_df = churn_df[churn_df.columns[1:]]
        self.cat_columns = cat_columns
        self.quant_columns = quant_columns
        self._is_encoded = False
        self._keep_columns = None
        self._response = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_data = None
        self.y_data = None
        self.rfc = None
        self.lrc = None


    def general_eda(self):
        '''
        perform basic eda and saves figures to ./images/eda folder
        '''
        churn_df = self.churn_df
        n_obs = churn_df.shape[0]
        n_features = churn_df.shape[1]
        print(
            f"The dataset has {n_features} features and {n_obs} observations.\n")

        cols_with_na = churn_df.columns[churn_df.isna().any()].tolist()
        if len(cols_with_na) == 0:
            print("There are no missing values in the dataset.\n")
        else:
            print("Columns with missing values, and missing entry fraction:")
            na_fraction = churn_df[cols_with_na].isnull().mean()
            na_fraction.name = "Missing entry fraction"
            print(na_fraction)
            print("\n")

        print("Basic statistical description:")
        print(churn_df.describe())
        print("\n")

        print("Correlation heatmap:")
        plt.figure(figsize=(20, 10))
        sns.heatmap(churn_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig('./images/eda/corr.png')

    def feature_dist(self, var):
        '''
        Displays and saves plot of a feature's distribution.
        input:
            var: string specifying the column used for plot.
        output:
            None
        '''
        churn_df = self.churn_df
        plt.figure(figsize=(20, 10))

        if var in self.cat_columns:
            churn_df[var].value_counts('normalize').plot(kind='bar')
        else:
            churn_df[var].hist()

        plt.savefig(f"./images/eda/histogram_{var}.png")

    def bivariate_dist(self, var):
        '''
        Displays and saves scatter plot of bivariate feature distribution.
        input:
            vars: list/tuple of feature strings with 2 entries.
        output:
            None
        '''
        churn_df = self.churn_df
        plt.figure(figsize=(20, 10))
        churn_df.plot.scatter(x=var[0], y=var[1])
        plt.savefig(f"./images/eda/bivariate_{var[0]}-{var[1]}.png")

    def encoder_helper(self, response="Churn"):
        '''
        helper function to create a churn column and use it to turn
        categorical columns into a new column with propotion of churn
        for each category

        input:
            response: string of response name [optional argument that
            could be used for naming variables or index y column; must not be in clumn names]
        output:
            None
        '''

        if self._is_encoded:
            print("Features have already been encoded")
        else:
            churn_df = self.churn_df
            churn_df[response] = churn_df["Attrition_Flag"].apply(
                lambda val: 0 if val == "Existing Customer" else 1)

            self._keep_columns = self.quant_columns

            for column in self.cat_columns:
                aux_lst = []
                cat_groups = churn_df.groupby(column).mean()[response]

                for val in churn_df[column]:
                    aux_lst.append(cat_groups.loc[val])
                new_col_name = f"{column}_{response}"
                churn_df[new_col_name] = aux_lst
                self._keep_columns.append(new_col_name)

            self.churn_df = churn_df
            self._response = response
            self._is_encoded = True

    def feature_engineering(self, test_fraction=0.3):
        '''
        input:
            response: string of response name [optional argument that could
            be used for naming variables or index y column]
            test_fraction: fraction of entries assigned to testing dataset.
        output:
            None
        '''
        x_data = self.churn_df[self._keep_columns]
        y_data = self.churn_df[self._response]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_data, y_data, test_size=test_fraction, random_state=42)
        self.x_data = x_data
        self.y_data = y_data

    def train_models(self,
                     n_estimators=(200, 500),
                     max_depth=(4, 5),
                     max_iter=1000
                     ):
        '''
        Module to train random forest (RF) and logistic regression  (logit) classification models.
        RF is trained across the parameters specified through inputs.

        inputs:
            n_estimators: list of int (RF)
            max_features: list of int, floats or a combination of "auto", "sqrt", "log2" (RF)
            max_depth: list of int (RF)
            max_iter: int (logit)
        output:
            None
        '''
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(max_iter=max_iter)

        param_grid = {
            "n_estimators": n_estimators,
            "max_features": ('auto', 'sqrt'),
            "max_depth": max_depth,
            "criterion": ["gini", "entropy"]
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.x_train, self.y_train)

        lrc.fit(self.x_train, self.y_train)

        self.rfc = cv_rfc.best_estimator_
        self.lrc = lrc

        joblib.dump(self.rfc, "./models/rfc_model.pkl")
        joblib.dump(self.lrc, "./models/lrc_model.pkl")

    def classification_report_image(self):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        '''
        y_train_preds_rf = self.rfc.predict(self.x_train)
        y_test_preds_rf = self.rfc.predict(self.x_test)

        y_train_preds_lr = self.lrc.predict(self.x_train)
        y_test_preds_lr = self.lrc.predict(self.x_test)

        plt.rc("figure", figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str("Random Forest Train"), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self.y_test, y_test_preds_rf)), {
                "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str("Random Forest Test"), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self.y_train, y_train_preds_rf)), {
                "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.savefig("./images/results/rfc_ClassificationReport.png")

        plt.rc("figure", figsize=(5, 5))
        plt.text(0.01, 1.25, str("Logistic Regression Train"),
                 {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    self.y_train, y_train_preds_lr)), {
                "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str("Logistic Regression Test"), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self.y_test, y_test_preds_lr)), {
                "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.savefig("./images/results/lrc_ClassificationReport.png")

        lrc_plot = plot_roc_curve(self.lrc, self.x_test, self.y_test)
        plt.figure(figsize=(15, 8))
        axis = plt.gca()
        rfc_disp = plot_roc_curve(
            self.rfc,
            self.x_test,
            self.y_test,
            ax=axis,
            alpha=0.8)
        lrc_plot.plot(ax=axis, alpha=0.8)
        plt.savefig("./images/results/ROC_AUC.png")

    def feature_importance_plot(self):
        '''
        creates and stores the feature importances per the RF model.
        '''

        importances = self.rfc.feature_importances_

        indices = np.argsort(importances)[::-1]

        names = [self.x_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(self.x_data.shape[1]), importances[indices])
        plt.xticks(range(self.x_data.shape[1]), names, rotation=90)
        plt.savefig("./images/results/rfc_feature_importances.png")

if __name__ == "__main__":
    PATH = "./data/bank_data.csv"
    churn_mdl = ChurnPredictor(PATH)
    churn_mdl.general_eda()
    churn_mdl.feature_dist("Card_Category")
    churn_mdl.feature_dist("Dependent_count")
    churn_mdl.bivariate_dist(["Customer_Age","Months_on_book"])
    churn_mdl.encoder_helper()
    churn_mdl.feature_engineering()
    churn_mdl.train_models()
    churn_mdl.classification_report_image()
    churn_mdl.feature_importance_plot()
