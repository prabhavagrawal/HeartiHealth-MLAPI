from flask import Flask,request,Response
app = Flask(__name__)

import json
import warnings
warnings.simplefilter("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.
df = pd.read_csv("heart.csv")

from sklearn.preprocessing import StandardScaler
# Import tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
# Define our features and labels
X = df.drop(['target'], axis=1).values
y = df['target'].values
#scale = StandardScaler()
#X = scale.fit_transform(X)
class Model:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        
        self.model.fit(self.X_train, self.y_train)
        print(f"{self.model_str()} Model Trained..")
        self.y_pred = self.model.predict(self.X_test)
        
    def model_str(self):
        return str(self.model.__class__.__name__)
    
    def crossValScore(self, cv=5):
        print(self.model_str() + "\n" + "="*60)
        scores = ["accuracy", "precision", "recall", "roc_auc"]
        for score in scores:  
            cv_acc = cross_val_score(self.model, 
                                     self.X_train, 
                                     self.y_train, 
                                     cv=cv, 
                                     scoring=score).mean()
            
            print("Model " + score + " : " + "%.3f" % cv_acc)
     
       
    def accuracy(self):
        accuarcy = accuracy_score(self.y_test, self.y_pred)
        print(self.model_str() + " Model " + "Accuracy is: ")
        return accuarcy
        
    def confusionMatrix(self):        
        plt.figure(figsize=(5, 5))
        mat = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(mat.T, square=True, 
                    annot=True, 
                    cbar=False, 
                    xticklabels=["Haven't Disease", "Have Disease"], 
                    yticklabels=["Haven't Disease", "Have Disease"])
        
        plt.title(self.model_str() + " Confusion Matrix")
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values');
        plt.show();
        
    def classificationReport(self):
        print(self.model_str() + " Classification Report" + "\n" + "="*60)
        print(classification_report(self.y_test, 
                                    self.y_pred, 
                                    target_names=['Non Disease', 'Disease']))
    
    def rocCurve(self):
        y_prob = self.model.predict_proba(self.X_test)[:,1]
        fpr, tpr, thr = roc_curve(self.y_test, y_prob)
        lw = 2
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 
                 color='darkorange', 
                 lw=lw, 
                 label="Curve Area = %0.3f" % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='green', 
                 lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.model_str() + ' Receiver Operating Characteristic Plot')
        plt.legend(loc="lower right")
        plt.show()
    
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction
    
    
@app.route('/')
def index():
    return "Service Running"

@app.route('/predict/',methods= ['POST'])
def predictor():

    try:
        
        req_data = request.get_json()

        InputData = np.array([req_data['age']])
        InputData = np.append(InputData,[req_data['sex']])
        InputData = np.append(InputData,[req_data['cp']])
        InputData = np.append(InputData,[req_data['trestbps']])
        InputData = np.append(InputData,[req_data['chol']])
        InputData = np.append(InputData,[req_data['fbs']])
        InputData = np.append(InputData,[req_data['restecg']])
        InputData = np.append(InputData,[req_data['thalach']])
        InputData = np.append(InputData,[req_data['exang']])
        InputData = np.append(InputData,[req_data['oldpeak']])
        InputData = np.append(InputData,[req_data['slope']])
        InputData = np.append(InputData,[req_data['ca']])
        InputData = np.append(InputData,[req_data['thal']])

        from sklearn.ensemble import RandomForestClassifier
        clf = Model(model=RandomForestClassifier(), X=X, y=y)
        clf.crossValScore(cv=10)
        clf.accuracy()
        df = np.array(InputData).reshape(1,-1)
        result = clf.predict(data=df)

        dict = {}
        dict['Result'] = result[0]
        dict['Status'] = 200

    except KeyError as e:
        
        dict = {}
        dict['Result'] = "Key missing " + str(e)
        dict['Status'] = 500

    except Exception as e:
        
        dict = {}
        dict['Result'] = "Something went wrong..!! " + str(e)
        dict['Status'] = 500

    response = str(dict)
    return response


@app.route('/accuracy/')
def accuracy():
    try:
        from sklearn.ensemble import RandomForestClassifier

        clf = Model(model=RandomForestClassifier(), X=X, y=y)
        clf.crossValScore(cv=10)
        clf.accuracy()

        dict = {}
        dict['Result'] = clf.accuracy()
        dict['Status'] = 200
        
    except Exception as e:
        
        dict = {}
        dict['Result'] = "Something went wrong..!! " + str(e)
        dict['Status'] = 500
    return str(dict)
