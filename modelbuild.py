import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



data = pd.read_csv("Crop_Recommendation.csv")

labels_encode = {
    'Rice':1, 
    'Maize':2, 
    'ChickPea':3, 
    'KidneyBeans':4, 
    'PigeonPeas':5,
    'MothBeans':6, 
    'MungBean':7, 
    'Blackgram':8, 
    'Lentil':9, 
    'Pomegranate':10,
    'Banana':11, 
    'Mango':12, 
    'Grapes':13, 
    'Watermelon':14, 
    'Muskmelon':15, 
    'Apple':16,
    'Orange':17, 
    'Papaya':18, 
    'Coconut':19, 
    'Cotton':20, 
    'Jute':21, 
    'Coffee':22
}

data["Crop"]=data["Crop"].map(labels_encode)

input = data.drop("Crop",axis=1)
output = data["Crop"]
x_train,x_test,y_train,y_test = train_test_split(input,output,test_size=0.2,random_state=1)
model_one = LogisticRegression()
model_one.fit(x_train,y_train)


def predict(nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall):
    result = model_one.predict([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
    return result