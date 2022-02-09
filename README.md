# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- By Diego Alfaro, 02-2022

## Project Description
This project contains basic routines to train models to diagnose/predict customer churn using two popular machine learning techniques for binary classification: logistic regression and random forests. Modeling and data manipulation relies mainly on Scikit-learn and Pandas libraries, and plotting is done via the matplotlib library. 

Routines are implemented as modules of a ChurnPredictor class defined in the churn_library.py file, including functionality for performing basic exploratory analysis of the data, encoding categorical variables, train-testing dataset spliting, model training and basic analysis of model performance and results. The implementation is based on the data in ./data/bank_data.csv, but it can be easily adapted for other csv datasets by specifying categorical and quantitative column names in the constants.py file (the only requirement is that the data contain an "Attrition_Flag" column with an "Existing Customer" category). 

## Requirements
Please refer to the requirements.txt file. To install all dependencies, on the command line from the main project fonder run

```console
pip install -r requirements.txt
```

## Running Files
The project includes the python script churn_library.py, which implements the ChurnPredictor class. This file can be run out-of-the-box on the command line with

```console
ipython churn_library.py
```

It will run all modules of the ChurnPredictor class, based on the data in ./data/bank_data.csv. In addition to the displayed output, models are saved to ./models and figures are saved to ./images. Please refer to code below the 'if __name__ == "__main__":' statement in churn_library.py.

The project also includes a testing file called churn_script_logging_and_tests_solution.py. To run the file, write the following on the command line:

```console
ipython churn_script_logging_and_tests_solution.py
```

Tests on each of ChurnPredictor's modules will be performed, and results are saved to a log file in logs/churn_library.log. Tests and expected results can be found below the 'if __name__ == "__main__":' statement within churn_script_logging_and_tests_solution.py. 

## The ChurnPredictor class

A ChurnPredictor class instance churn_predictor is created with churn_predictor = ChurnPredictor(PATH), where PATH specifies path to data. Use churn_predictor.churn_df attribure to access dataframe. The class includes the following modules:

* general_eda(): Performs basic exploratory data analysis, displaying basic numeric info and saving a correlation plot to ./images/eda
* feature_dist(var_name): Displays a histogram plot of column var_name and saves figure to ./images/eda
* bivariate_dist((var_name1, var_name2)): Displays a scatter-plot of observations on the plane var_name1-var_name2 and saves fiture to ./images/eda
* encoder_helper(response): Encodes categorical variables, including the dependent variable in "Attrition_Flag" and renames it as response (defaults to "churn")
* feature_engineering(test_fraction): Splits dataset into training and testing datasets per the spedified test_fraction. Results are in attributes x_train, x_test, y_train, y_test. Must run encoder_helper first. 
* train_models(n_estimators, max_depth, max_iter): Trains a random forest classifier (rf) and a logistic regression classifier (lrc), the models being stored in attributes rfc and lrc, respectively, and saved to ./models. The rfc is trained by performing scikit-learn's GridSearchCV on values n_estimators and max_depth. max_iter specifies the maximum number of iterations to fit lrc. Refer to https://scikit-learn.org/ documentation for more information.
* classification_report_image(): Creates a classification report for each model. It also displays ROC curves and AUC values. Report and ROC curve mages are saved to ./images/results
* feature_importance_plot(): Displays an rfc feature importance plot, and saves to ./images/results
