import numpy as np
import pandas as pd
import pickle


def getEncoded(test_data,labelencoder_dict,onehotencoder_dict):
    test_encoded_x = None
    for i in range(0,test_data.shape[1]):
        label_encoder =  labelencoder_dict[i]
        feature = label_encoder.transform(test_data[:,i])
        feature = feature.reshape(test_data.shape[0], 1)
        onehot_encoder = onehotencoder_dict[i]
        feature = onehot_encoder.transform(feature)
        if test_encoded_x is None:
          test_encoded_x = feature
        else:
          test_encoded_x = np.concatenate((test_encoded_x, feature), axis=1)
    print(test_encoded_x)
    return test_encoded_x


loading_data=pd.read_csv("/home/vishal/MEGA/ClassificationProblems/oneHotEncodingDecoding/dataSet/test.csv",sep=",")
feature_selection = loading_data[['Name', 'Country']].values
test_data = feature_selection.astype('str')

load_labelencoder_dict=np.load("labelencoder_dict.npy")
load_onehotencoder_dict=np.load("onehotencoder_dict.npy")

labelencoder_dict = dict(load_labelencoder_dict.tolist())
onehotencoder_dict = dict(load_onehotencoder_dict.tolist())


model = pickle.load(open("oneHotEncodingDecoding.sav", 'rb'))
prediction_data=getEncoded(test_data,labelencoder_dict,onehotencoder_dict)


print(model.predict(prediction_data))

