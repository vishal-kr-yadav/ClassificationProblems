import ConfigParser
import pandas as pd
from sklearn import metrics



Config = ConfigParser.ConfigParser()
Config.read("/home/vishal/MEGA/ClassificationProblems/modelSelection/modelSelection.ini")
data_set_path=Config.get('General','data_set_path')
file_separator=Config.get('General','file_separator')
independent_var_axis=Config.get('General','independent_var_axis')
dependent_var_axis=Config.get('General','dependent_var_axis')
train_size=Config.get('General','train_size')




dataSet=pd.read_csv(data_set_path,sep=file_separator)

# silicing
x=dataSet.iloc[:,int(independent_var_axis[0]):int(independent_var_axis[2])].values
y=dataSet.iloc[:,int(dependent_var_axis)].values


# splitting the dataset in train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=float(train_size),random_state=0)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)

# Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

#KNN - K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

# Stochastic Gradient Descent
from sklearn import linear_model
clf = linear_model.SGDClassifier()
clf.fit(x_train, y_train)
acc_sgd = round(clf.score(x_train, y_train) * 100, 2)


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
print(models)







