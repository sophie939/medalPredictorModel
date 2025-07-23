# medalPredictorModel

# Olympic Medal Prediction using Gradient Boosting 

---

## Project Overview

I created this project to practice implementing an end-to-end machine learning pipeline. I used a SQLite relational database on DB Browser that I had created previously for a university course that stores Olympic results. It is strucutred by personal athelete information, sport details and country socio-economic data for the 2008, 2012, 2016 and 2020 Summer Olympic Games. It was created, cleaned and structured by myself and three other classmates. The raw data was primarily sourced from this Github repository: https://github.com/KeithGalli/Olympics-Dataset/tree/master

The goal is to predict whether an athlete will win a medal (Gold, Silver, or Bronze) using:

- **Physical attributes**: Height, weight, health metrics
- **Economic data by country**: GDP, GDP per capita, population
- **Historical performance**: Country and sport success rates
- **Additional Calulated Features**: BMI, performance ratios

I considered multiple classification algorithms to implement including logistic regression and random forests, but landed on gradient boosting for this version.

The repository includes the following files:

- olympic_ml_notebook.ipynb
-     Imports, database connection, summary statistics, feature engineering, model developement and testing.
- olympic_medal_predictor_gb.pkl
-     Stores the model that has just created to avoid re-training.
- model_metadata.pkl
-     Stores key information about the model including test results, hyperparameters and training data size.

---
