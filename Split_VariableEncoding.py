import pandas as pd
mydataset1 = pd.read_csv('Hotel.csv')
X = mydataset1.iloc[:,:-1].values
Y = mydataset1.iloc[:,4].values
Z = pd.DataFrame(X)
W = pd.DataFrame(Y)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
Y = labelencoder_X.fit_transform(Y)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
Z_train = pd.DataFrame(X_train)
Z_test = pd.DataFrame(X_test)
W_train = pd.DataFrame(Y_train)
W_test = pd.DataFrame(Y_test)