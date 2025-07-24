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
-   Synthetic data using Python 'random' module: heart rate variability, vo2Max, blood oxygen levels
- **Economic data by country**:
- Derived values: GDP per capita
- **Historical performance**:
-  Derived values: Country medal rate, average country ranking, sport medal rate, sport average ranking

I considered multiple binary classification algorithms to implement including logistic regression and random forests, but landed on gradient boosting for this version. To test the accuracy of my baseline model, I used a 80/20 test-train split with five-fold cross-validation. My mean Area Under the Curve Reciever Operator Characteristic (AUC ROC) score on the training data was 0.7506 suggesting that my model correctly discriminates between athletes who win a medal and who don't 75.06% of the time. Low standard deviation reassured me that the model is consistent across different subsets of data. To find the optimal choice of hyperparamters for my model, I compared the AUC-ROC score for each combination of the following hyperparaters used in gradient boosing: number of trees, maximum depth of each tree, the learning rate and the minimum number of samples required to split an internal node. This improved the AUC-ROC score by 0.72%.

When evaluating my model on the test data, it had high accuracy as it correctly classified 89.63% of test cases. However, the majority of these cases were for correctly classifyig non-medal winners. A more meaningful metric would be the recal since the data set is imbalanced and the number of true positives (medal winners) is the more important category and it is very low at only 10.8%. The model has a recal of only 9%, meaning out of all medal winners within the test data, it correctly classifies 9%. However, this could be due to the test data size as only 462/4280 athletes were medal winners. The recall for non-medal winners was very high at 99%, so this cateogry is clearly easier for the model to identify. The model has good precision, as when it predicts a medal winner it is correct 64% of the time. The trade-off is that this is not predicting a medal winner frequently enough. Re-evaluating the model with a larger test/train split may be required.

The most predictive aspects of whether an athlete wins a medal were the country and sport medal rate respectively, suggesting that understanding the wider picture is important when assessing athlete performance since these metrics were almost an order of mangnitiude more predictive than physical characteristics. This could be explained by access to experienced coaches and a larger number of athletes to train with if a particular country is successful. Certain sports will have more medals available to them such as rowing and swimming, causing bias in this model. A more accurate version would analyse the medal rates within specific events. This allows the model to use the prior information of the medal-frequency within the sport before generating a prediction. Finally, the physical attributes has not taken into account gender, meaning that the weights of a 'fitter' athlete are not adjusted to their gender.
The repository includes the following files:

-     olympic_ml_notebook.ipynb
  Imports, database connection, summary statistics, feature engineering, model developement and testing.
-     olympic_medal_predictor_gb.pkl
  Stores the model that has just created to avoid re-training.
-     model_metadata.pkl
  Stores key information about the model including test results, hyperparameters and training data size.

---
