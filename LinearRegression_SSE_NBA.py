#Default imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading the file and assigning columns to variables
data = pd.read_csv('NBA Players.csv',encoding='ISO-8859-1')
heights = data['Height']
weights = data['Weight']

#Calulating mean
avg_height = heights.mean()
avg_weight = weights.mean()

#Plotting a scatter plot to get an overview of the weight and height
plt.scatter(weights,heights)
plt.xlabel('Weights')
plt.ylabel('Heights')
plt.show

#Taking correlation coefficient between two variables to understand the correlation between the two
corr_coeff = np.corrcoef(weights,heights)

# Here we are calculating b1 = slope
deviation_products_list = [(w - avg_weight)*(h - avg_height) for h,w in zip(heights,weights)]
weights_deviation_list = [(w-avg_weight)**2 for w in weights]
deviation_products = sum(deviation_products_list)
weights_deviation = sum(weights_deviation_list)
slope = deviation_products / weights_deviation

#calulating b0=line of intersection(y_inter)
y_inter = avg_height - (slope*avg_weight)

#Creating a function to predict height using weight
def predict_height(weight1):
    return y_inter + (slope*weight1)

predict_height(263)