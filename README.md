# Road Accident Severity Prediction

## Project Overview

Road Accident Severity Prediction is a Machine Learning classification project that predicts the severity of a road accident as **Fatal, Serious, or Slight** based on accident-related features.

The project uses an **Artificial Neural Network (ANN)** for classification and includes data preprocessing, categorical encoding, feature scaling, class imbalance handling, model training, evaluation, and deployment using Streamlit.

## Problem Statement

Road accidents can result in different levels of severity depending on various factors such as road conditions, vehicles involved, environmental conditions, and other accident-related characteristics.

The objective of this project is to build a Machine Learning model that can predict accident severity and provide an interactive application for making predictions.

## Objectives

* Analyze road accident data
* Perform data cleaning and preprocessing
* Handle missing values and duplicate records
* Encode categorical features
* Scale numerical features
* Handle class imbalance
* Build an Artificial Neural Network classification model
* Evaluate the trained model
* Deploy the model using Streamlit
* Provide real-time accident severity predictions

## Dataset

The dataset contains road accident records with multiple accident-related attributes.

The target variable represents accident severity with three classes:

* Fatal
* Serious
* Slight

The project works with a large accident dataset containing approximately **307,000 records**.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* Streamlit
* Matplotlib
* Seaborn
* Pickle
* Jupyter Notebook

## Machine Learning Workflow

The project follows the following workflow:

1. Problem Understanding
2. Data Loading
3. Data Understanding
4. Exploratory Data Analysis
5. Data Cleaning
6. Missing Value Analysis
7. Duplicate Handling
8. Feature Engineering
9. Feature Encoding
10. Feature Scaling
11. Class Imbalance Handling
12. Train-Test Split
13. ANN Model Building
14. Model Training
15. Model Evaluation
16. Model Saving
17. Streamlit Deployment

## Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values using `SimpleImputer`
* Encoding categorical variables using `OneHotEncoder`
* Encoding ordinal variables using `OrdinalEncoder`
* Scaling numerical features using `StandardScaler`
* Handling class imbalance using class weights

The preprocessing components are saved using Pickle so that the same transformations can be applied during prediction.

## Artificial Neural Network

An Artificial Neural Network (ANN) is used as the final classification model.

The trained ANN learns patterns from the preprocessed accident data and predicts one of the three accident severity classes:

```text
Fatal
Serious
Slight
```

Early stopping and class weights are used during training to improve model training and address class imbalance.

## Model Evaluation

The model is evaluated using classification-related performance metrics.

The evaluation process includes:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

These metrics help evaluate how effectively the model predicts each accident severity class.

## Model Deployment

The trained ANN model is deployed using **Streamlit**.

The Streamlit application:

1. Accepts accident-related input from the user
2. Applies the saved preprocessing pipeline
3. Passes the processed input to the ANN model
4. Predicts the accident severity
5. Converts the predicted class back to its original label

The deployment uses the following saved files:

```text
architechture.h5
preprocessor.pkl
label_encoder.pkl
```

## Project Structure

```text
Road-Accident-Severity-Prediction/
│
├── Road_Accident_Severity_Prediction.ipynb
├── app.py
├── preprocessor.pkl
├── label_encoder.pkl
├── architechture.h5
├── requirements.txt
└── README.md
```

## Streamlit Application

To run the application locally, install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser and allow users to enter accident-related information and obtain the predicted accident severity.

## Key Learnings

Through this project, I gained hands-on experience in:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Handling missing values
* Categorical feature encoding
* Numerical feature scaling
* Handling imbalanced classification
* Artificial Neural Networks
* TensorFlow and Keras
* Model evaluation
* Model serialization
* Streamlit application development
* Machine Learning model deployment

## Future Improvements

Possible improvements include:

* Experimenting with additional Machine Learning and Deep Learning models
* Further feature engineering
* Hyperparameter optimization
* Improving minority-class prediction
* Adding additional evaluation visualizations
* Improving the Streamlit user interface
* Continuous monitoring of model performance after deployment

## Conclusion

This project demonstrates the development of an end-to-end Machine Learning solution for road accident severity prediction using an Artificial Neural Network.

It covers the complete workflow from data preprocessing and model training to evaluation and deployment, providing practical experience in Machine Learning classification and application deployment.

## Author

**Bhuvana Mandli**

Machine Learning and Data Science Learner
