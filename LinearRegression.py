import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#Reading the file and assigning columns to variables
mydataset1 = pd.read_csv('Revenue.csv',index_col=0)
X = mydataset1.iloc[:,:-1].values
Y = mydataset1.iloc[:,1].values

#Splitting the dataset into train(80%) and test(20%)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#Creating an object
regressor = LinearRegression()

#Implementing Linear Regression
regressor.fit(X_train,Y_train)


Y_pred = regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Revenue vs Reviews (Training Set)')
plt.xlabel('Customer Reviews')
plt.ylabel('Revenue')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Revenue vs Reviews (Training Set)')
plt.xlabel('Customer Reviews')
plt.ylabel('Revenue')
plt.show() 