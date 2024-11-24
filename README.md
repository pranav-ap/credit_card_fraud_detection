# Credit Card Fraud Detection

In this project, the goal is to classify credit card transactions as either 
fraudulent or legitimate, based on the given features. An XGBoost Classifier was trained to 
do this.   

The dataset used in this project is from Kaggle and can 
be found [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data). 
It contains credit card transactions made by European cardholders in the year 2023. There are 
over 550,000 anonymized records.

Initially, in this project, PCA was used to reduce the dimensionality of the data to around 
22 columns. This covered around 95% of the variance in the data. The skew in the data was 
then reduced using the `yeo-johnson` power transformation. 

The XGBoost Classifier was then trained on this transformed dataset. The PR Curve, ROC Curve,
and confusion matrix were used to evaluate the model. They can be found in the 
training notebook.


