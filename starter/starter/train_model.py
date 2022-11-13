import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model,compute_model_metrics,inference,slice_metrics
import pickle


data =pd.read_csv('../data/census.csv')
train, test = train_test_split(data, test_size=0.20,stratify=data["salary"])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model=train_model(X_train, y_train)

with open('../model/rfc_model.pkl', 'wb') as files:
    pickle.dump(model, files)

with open('../model/lb.pkl', 'wb') as files:
    pickle.dump(lb, files)

with open('../model/encoder.pkl', 'wb') as files:
    pickle.dump(encoder, files)


X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder=encoder,lb=lb
)
test_prediction =inference(model, X_test)
print(compute_model_metrics(y_test, test_prediction))
# test['prediction'] =test_prediction
# print(test[test['prediction']==1])
# test[test['prediction']==1].to_csv("prediction.csv",index=False)

slice_metricses={}
for slice_val in np.unique(test['education']):
    precision, recall, fbeta=slice_metrics(model,test, slice_col="education",slice_val=slice_val,cat_features=cat_features,encoder=encoder,lb=lb)
    f = open("../data/slice_output.txt", "a")
    f.write("\nslice: {} precision : {} recall: {} fbeta: {}\n".format(slice_val,precision,recall,fbeta))
    f.close()

