import ConfigParser
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle

from sklearn import metrics



Config = ConfigParser.ConfigParser()
Config.read("model.ini")
model_name=Config.get('General','model_name')
model_save=Config.get('General','model_save')



data_set_path=Config.get('General','data_set_path')
file_separator=Config.get('General','file_separator')
independent_var_axis=Config.get('General','independent_var_axis')
dependent_var_axis=Config.get('General','dependent_var_axis')
train_size=Config.get('General','train_size')



filename = model_save+'.sav'

dataSet=pd.read_csv(data_set_path,sep=file_separator)

# silicing the inde
x=dataSet.iloc[:,int(independent_var_axis[0]):int(independent_var_axis[2])].values
y=dataSet.iloc[:,int(dependent_var_axis)].values


# splitting the dataset in train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=float(train_size),random_state=0)



def logisticRegression(x_train,y_train):
    logreg=LogisticRegression()
    model=logreg.fit(x_train, y_train)
    # for prediction on a complete file, replace below  line with model.predict(x)
    # print(model.predict([[49 ,28000]]))

    pickle.dump(model, open(filename, 'wb'))

def svc(x_train,y_train):
    linear_svc = LinearSVC()
    model=linear_svc.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def knn(x_train,y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    model=knn.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def gaussian(x_train,y_train):
    gaussian = GaussianNB()
    model=gaussian.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def perceptron(x_train,y_train):
    perceptron = Perceptron()
    model=perceptron.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def SGDClassifier(x_train,y_train):
    clf = linear_model.SGDClassifier()
    model=clf.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def decisionTree(x_train,y_train):
    decision_tree = DecisionTreeClassifier()
    model=decision_tree.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))

def randomForest(x_train,y_train):
    random_forest = RandomForestClassifier(n_estimators=100)
    model=random_forest.fit(x_train, y_train)
    pickle.dump(model, open(filename, 'wb'))



if model_name=="logisticRegression":
    logisticRegression(x_train,y_train)
elif model_name=="svc":
    svc(x_train,y_train)
elif model_name=="knn":
    knn(x_train,y_train)
elif model_name=="gaussian":
    gaussian(x_train,y_train)
elif model_name=="perceptron":
    perceptron(x_train,y_train)
elif model_name=="SGDClassifier":
    SGDClassifier(x_train,y_train)
elif model_name=="decisionTree":
    decisionTree(x_train,y_train)
elif model_name=="randomForest":
    randomForest(x_train,y_train)
else:
    print("Please enter the correct model name")