# Sparks-ML-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sb
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
# reading data from the link and displaying first 10 entries 
fd=pd.read_csv('http://bit.ly/w-data')
fd.head(5)
fd.shape #to check the number rows and columns 
# dividing the data into attributes and labels
X = fd.iloc[:, :-1].values  
y = fd.iloc[:, 1].values  
# splitting the data into 2 train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# plotting bar graph
x=fd.sample(5)
x.plot.bar()
# training the algorithm
# linear regression
lr = LinearRegression()  
lr.fit(X_train, y_train) 
lr.score(X_test, y_test)
# plotting regression line 
ax=fd['Hours']
ay=fd['Scores']
sb.regplot(ax,ay,color='b',data=fd)
# prediction over train set and calculating error 
train_predict=lr.predict(X_train)
k=mae(train_predict,y_train)
print('TRAINING MEAN ABSOLUTE ERROR',k)
#prediction over test set and calculating error 
test_predict=lr.predict(X_test)
k=mae(test_predict,y_test)
print('TESTING MEAN ABSOLUTE ERROR',k)
lr.score(X_train,y_train),lr.score(X_test,y_test)
# Comparing Actual vs Predicted
pd.DataFrame({'Actual': y_test, 'Predicted': test_predict})  
# You can also test with your own data
#predicting the score by giving the hours as an input
hours = [[2.6]]
predict = lr.predict(hours)
print("No of Hours = {}".format(hours[0]))
print("Predicted Score = {}".format(predict[0]))
