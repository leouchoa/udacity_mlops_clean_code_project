# Predict Customer Churn

This is my solution to the **Predict Customer Churn** Project, of the [ML DevOps Engineer Nanodegree Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). This aims to:

- Write a ML pipeline to predict churn, using [this](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code) dataset
- Make the code reusable, by following some code best practices.

## Coding Best Practices

Employing some code organization and readability can greatly improve our quality of life while coding, making it more pleasant. The idea is that early directories/codes structuring will make us work more at the beginning of the project, but in the long run will cause us to spend significant less time reducing [technical debt](https://www.productplan.com/glossary/technical-debt/). Here is a short list of its benefits:

- automated unit/integration function testing allows faster debugging.
- self-explanatory code makes us quickly recap what was done and collaborate easier.
- automated pipelines provides faster/safer ml models retraining. 



## Running Files

There are two main files to be run: 

- `churn_library.py`: the file that runs all the model creating and report pipeline.
- `churn_script_logging_and_tests.py`: tests if the functions created in `churn_library.py` behave as intended to.

To run the **first**, please use

```
python churn_library.py
```

To run the **second**, you either run

```
python churn_script_logging_and_tests.py
```

or 

```
pytest churn_script_logging_and_tests.py
```

## Expected Result

For `python churn_library.py` you can expect your directory to be populated as in the tree bellow, while for `pytest churn_script_logging_and_tests.py` the expected result will be very similar to what the `pytest.log` file depicts.

```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transaction_distribution.png
│   └── results
│       ├── feature_importance_shap.png
│       ├── feature_importances.png
│       ├── logistic_results.png
│       └── rf_results.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library.cpython-36.pyc
│   ├── churn_library.cpython-38.pyc
│   └── churn_script_logging_and_tests.cpython-38-pytest-6.2.5.pyc
└── README.md

7 directories, 21 files
```
