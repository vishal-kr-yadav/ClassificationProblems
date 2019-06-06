import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle

# data = [['Alex','USA1',1],['Bob','India1',0],['Bob','India',1],['Alex','USA',0],['Clarke','SriLanka1',1],['Alex','USA',0],['Alex','USA',0]]

# df = pd.DataFrame(data,columns=['Name','Country','Traget'])
df=pd.read_csv("/home/vishal/MEGA/ClassificationProblems/oneHotEncodingDecoding/dataSet/train.csv",sep=",")
Y_train=df["Target"].values
# print(Y_train)
# X=df[["Name","Country"]]
X = df[['Name', 'Country']].values
X = X.astype('str')
# print(X)
labelencoder_dict = {}
onehotencoder_dict = {}
X_train = None
for i in range(0, X.shape[1]):
    label_encoder = LabelEncoder()
    labelencoder_dict[i] = label_encoder
    feature = label_encoder.fit_transform(X[:,i])
    feature = feature.reshape(X.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    onehotencoder_dict[i] = onehot_encoder
    if X_train is None:
      X_train = feature
    else:
      X_train = np.concatenate((X_train, feature), axis=1)

# print(X_train)
# splitting the dataset in train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,train_size=float(0.5),random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
model=logreg.fit(x_train, y_train)





acc_log = round(model.score(x_train, y_train) * 100, 2)
print(acc_log)

np.save('labelencoder_dict.npy', labelencoder_dict)
np.save('onehotencoder_dict.npy',onehotencoder_dict)
pickle.dump(model, open("oneHotEncodingDecoding.sav", 'wb'))