#pandas used for representing/manipulating arrays
import pandas as pd
import numpy as np
#allows us to split our data into a group used to train alg and test group
from sklearn.model_selection import train_test_split
#preprocess, or make normalizations to data before model is fit
from sklearn import preprocessing
#random forest family of models is imported
from sklearn.ensemble import RandomForestRegressor
#tools for cross-validation, or using training set to improve the model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#allows us to evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
#allows us to save model and use it in the future.
from sklearn.externals import joblib

#Loading/viewing data:
dataURL = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataURL, sep = ";")
#prints a list of features and first 4 samples and feature values:
#print(data.head())
#prints the shape of data, or (samples, features)
#print(data.shape)
#prints stats about data for each feature
#print(data.describe())

#Splitting data:
#We make the target variable y be the feature quality we are trying to predict
y = data.quality
#make X the data without the target feature, axis = 1 specifies it is a column
X = data.drop('quality', axis = 1)
#now, we want to split the data into train and test data
#20% of data will be the test data. we assign the split a random state so we can reproduce our results.
#We stratify data according to target variable to make sure no subgroups of the data
#are notably different- we don't want data that is not represented equally
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123, stratify = y)

#Let's preprocess!
'''
We could scale data set by simply using z-score with training data. 
However, this would not work because the test data would have a different mean and SD,
so the results would not be accurate.
Instead, let's use the Transformer API so we ca apply it to future data sets.
'''
scaler = preprocessing.StandardScaler().fit(X_train)
Xtrain_scaled = preprocessing.scale(X_train)
#using same scaler object to transform the test data
Xtest_scaled = scaler.transform(X_test)
#let's make a pipeline and adjust hyperparameters in order to create the best model possible.
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100))
#hyperparameters are the structural options of the model
#model parameters are what we learn from data
#We can use a python dictionary for the model to try the options 
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
'''
Cross-Validation:
We use cross validation to maximize performance without overfitting.
We use the training data itself the test the model.
Split the data into folds, reserving one fold for testing. Use the rest of the 
folds to train the model.
Repeat this k times, holding out a different fold every time.
GridSearchCV performs cross validation with specified model & # of folds.
Then, a small performance improvement is achieved by using the whole training data
through the refit functionality.
The optimized model results.
'''
clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
#use data to fit model.
clf.fit(X_train, y_train)

'''
Evaluate model!
'''
y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

'''
save model for future use.
to load in future:
future_clf = joblib.load('rf_regressor.pkl')
'''
joblib.dump(clf, 'rf_regressor.pkl')


