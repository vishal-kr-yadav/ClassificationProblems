import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

#Prediction using the save model
model = pickle.load(open("finalized_model.sav", 'rb'))

# age and EstimatedSalary
print(model.predict([[49 ,28000]]))

# result=pd.DataFrame(predicted_value,columns=['PredictedValue'])
# df_with_predicted_col = pd.concat([predict_dataSet, result],axis=1)


# df_with_predicted_col.to_csv('prediction.csv',sep=',')
