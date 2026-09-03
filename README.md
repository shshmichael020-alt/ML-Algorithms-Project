# ML-Algorithms-Project

## Machine Learning for Network Traffic Analysis

### About the Project

This project focuses on the practical implementation and evaluation of different machine learning algorithms for network traffic analysis. Network traffic datasets contain a large number of features that can be used to study network behaviour and identify different types of traffic and attacks.

The project uses network traffic data based on the CICIDS dataset and applies different machine learning techniques to the data. The main purpose is to understand how different algorithms work on the same dataset, how the data needs to be prepared before training, and how model performance can be evaluated and compared.

The project implements five machine learning algorithms:

- Linear Regression
- Logistic Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

The project is being developed collaboratively, with different stages of implementation being completed by different team members.

---

## Project Objective

The main objectives of this project are:

- To understand the practical application of machine learning algorithms.
- To work with a large real-world network traffic dataset.
- To perform data preprocessing and prepare the dataset for machine learning.
- To understand the difference between classification and regression approaches.
- To train different machine learning models using the same dataset.
- To evaluate model performance using suitable evaluation metrics.
- To compare the behaviour and performance of different algorithms.
- To gain practical experience with Python and the Scikit-learn library.
- To understand the importance of regularization in regression models.
- To gain experience in managing and developing a collaborative machine learning project using Git and GitHub.

---

## Dataset

The project uses network traffic data based on the CICIDS dataset.

The dataset contains network-flow information with multiple numerical features describing network connections and traffic behaviour. The target column represents the corresponding traffic or attack category.

The dataset is useful for studying machine learning applications in areas such as:

- Network security
- Intrusion detection
- Traffic classification
- Cybersecurity analytics
- Anomaly and attack detection

The dataset files are kept locally because of their large size and are not included in this GitHub repository.

---

## Data Preprocessing

Data preprocessing is an important part of this project because machine learning algorithms require clean and suitable input data.

The following preprocessing steps were performed:

### 1. Combining the datasets

Multiple network traffic CSV files were loaded and combined into a single dataset so that the models could be trained and tested using a larger and more representative collection of network traffic.

### 2. Cleaning column names

The column names were stripped of unnecessary spaces to ensure consistent feature names during processing.

### 3. Converting features to numerical values

The input features were converted into numerical form using Pandas. Values that could not be converted into numbers were treated as missing values.

### 4. Handling infinite values

Infinite and negative-infinite values were replaced with missing values because such values can cause problems during model training.

### 5. Handling missing values

Missing values were handled using mean imputation.

The mean value of the available data for a feature was used to replace missing values, allowing the dataset to be used by the machine learning models without removing large numbers of records.

### 6. Encoding the target variable

The target labels were categorical, so Label Encoding was used to convert the categories into numerical values that could be processed by the machine learning algorithms.

### 7. Splitting the dataset

The complete dataset was divided into:

- 80% training data
- 20% testing data

A stratified split was used so that the distribution of target classes was maintained as much as possible in both training and testing sets.

---

## Algorithms Implemented

### 1. Linear Regression

Linear Regression is a supervised learning algorithm that models the relationship between input features and a target variable using a linear equation.

In this project, Linear Regression was implemented to understand how a basic regression model behaves when applied to the encoded network traffic target.

The model was trained using the training dataset and its predictions were evaluated using regression metrics.

---

### 2. Logistic Regression

Logistic Regression is a supervised classification algorithm commonly used when the target variable consists of different classes.

In this project, Logistic Regression was used to classify network traffic into the encoded target categories.

The model produces class predictions that can be evaluated using classification metrics such as accuracy, precision, recall, F1 score, and a confusion matrix.

---

### 3. Ridge Regression

Ridge Regression is an extension of Linear Regression that uses L2 regularization.

Regularization adds a penalty to large model coefficients and helps reduce the possibility of overfitting.

The Ridge model was implemented with:

```text
alpha = 1.0