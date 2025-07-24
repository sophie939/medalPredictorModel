# medalPredictorModel

# Olympic Medal Prediction using Gradient Boosting 

---

## Overview

I created this project to practice implementing an end-to-end machine learning pipeline. I used a SQLite relational database on DB Browser that I had created previously for a university course that stores Olympic results. It is strucutred by personal athelete information, sport details and country socio-economic data for the 2008, 2012, 2016 and 2020 Summer Olympic Games. It was created, cleaned and structured by myself and three other classmates. The raw data was primarily sourced from this Github repository: https://github.com/KeithGalli/Olympics-Dataset/tree/master

The goal is to predict whether an athlete will win a medal (Gold, Silver, or Bronze) in comparison to the average medal rate from 2008-2020 of 10.8%. The database contains 34 features across 11 tables with 21,398 records spanning 12,354 athletes, 176 countries and 48 sports. After data cleaning, the total final data set contained 17,118 records. 

Out of the initial features, I narrowed down my model to focus on the 16 features that were grouped by the following categories:

- **Physical attributes**:
-   Height, weight
-   Derived values: BMI, a general health score and an injury risk score
-   Synthetic data using Python 'random' module: heart rate variability (synthetic), vo2Max (synthetic), blood oxygen levels (synthetic), 
- **Economic data by country**:
- Derived values: GDP per capita
- **Historical performance**:
-  Derived values: Country medal rate, average country ranking, sport medal rate, sport average ranking

I considered multiple classification algorithms to implement including logistic regression and random forests, but landed on gradient boosting for this version.

The repository includes the following files:

-     olympic_ml_notebook.ipynb
  Imports, database connection, summary statistics, feature engineering, model developement and testing.
-     olympic_medal_predictor_gb.pkl
  Stores the model that has just created to avoid re-training.
-     model_metadata.pkl
  Stores key information about the model including test results, hyperparameters and training data size.

---
