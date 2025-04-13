# Predicting-Purchase-Intent-Ecommerce
Project Overview
This project involves predicting purchase intent for an e-commerce platform using machine learning models. The goal is to analyze user behavior and predict whether a user will make a purchase based on various features, including interaction duration, page views, and other user characteristics.

### Techniques Used:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

### Data Description
The dataset used in this project is the Online Shoppers Intention Dataset fro UCI Machine Learning Repository. It contains information about user sessions on an e-commerce website, with the target variable being whether or not a user made a purchase.

### Key Features:
1. Administrative Duration: Time spent on administrative pages.
2. ProductRelated Duration: Time spent on product-related pages.
3. Informational Duration: Time spent on informational pages.
4. Bounce Rate: Percentage of sessions where users leave after viewing only one page.
5. Exit Rate: Percentage of sessions that end on a given page.
6. Revenue: Target variable indicating whether the user made a purchase (0 = No, 1 = Yes).

### Objectives
Objective 1: Analyze the influence of page interaction durations on user purchase behavior.

Objective 2: Evaluate the effect of bounce rate and exit rate on conversion likelihood across sessions.

Objective 3: Compare purchasing behavior across months and special days to identify seasonal trends.

Objective 4: Investigate the relationship between visitor type and purchasing behavior.

Objective 5: Examine regional and traffic type variations in user purchase behavior.

### Data Preprocessing
Before applying machine learning models, the following preprocessing steps were performed:
1. Removed duplicate rows from the dataset.
2. Filtered sessions where all durations (administrative, informational, product-related) were zero.
3. Handled missing values using forward fill (for year columns).
4. Encoded categorical variables (VisitorType and Month) using one-hot encoding.
5. Scaled the features using MinMaxScaler to normalize the dataset and improve model performance.

### Machine Learning Models
The project applies three different machine learning models to predict purchase intent:

1. Logistic Regression: A binary classification model used to predict the likelihood of a user making a purchase.
2. K-Nearest Neighbors (KNN): A non-parametric method used for classification by finding the closest neighbors to a data point.
3. Random Forest Classifier: An ensemble learning method that uses multiple decision trees to predict the target variable.

Each model is evaluated using accuracy and confusion matrix metrics.

### Model Evaluation
The models were evaluated on their accuracy using the test set, with a focus on:

1. Logistic Regression: ~80% accuracy
2. KNN Classifier: ~84% accuracy
3. Random Forest: ~86% accuracy

Additionally, confusion matrices were used to assess the performance and identify misclassifications.


### Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Contributions, suggestions, and feedback are always welcome!
