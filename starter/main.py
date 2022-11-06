from turtle import st
from fastapi import FastAPI, Request
import pickle
from pydantic import BaseModel,Field
from starter.ml.data import process_data
import pickle
import pandas as pd

# model = pickle.load(open('./model/rfc_model.pkl', 'rb'))
# lb = pickle.load(open('./model/lb', 'rb'))
# encoder = pickle.load(open('./model/encoder', 'rb'))
# cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

class Taggeditem(BaseModel):
    age: float =Field(example=45)
    workclass: str =Field(example="State-gov")
    fnlgt: float =Field(example=77516)
    education: str =Field(example="Bachelors")
    education_num: float = Field(example=13,alias="education-num")
    marital_status: str = Field(example="Never-married",alias="marital-status")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str =Field(example="White")
    sex: str =Field(example="Male")
    capital_gain: float = Field(example=2174,alias='capital-gain')
    capital_loss: float =Field(example=0,alias='capital-loss')
    hours_per_week: float =Field(example=40,alias='hours-per-week')
    native_country: str =Field(example="United-States",alias='native-country')




app = FastAPI()

@app.get("/") ## define an endpoint
async def greeting():
    return {"Greetings": "Welcome to the MLops World!"}


@app.post("/inference/")
async def inference(request: Taggeditem): #,body=Taggeditem

    data =  request.dict()
    data1={}
    for key in data.keys():
        data1[key] =[data[key]]
    data = pd.DataFrame(data1)
    data.columns =[col.replace("_","-") for col in data.columns]

    model = pickle.load(open('/home/purusanth/project3/nd0821-c3-starter-code/starter/model/rfc_model.pkl', 'rb'))
    lb = pickle.load(open('/home/purusanth/project3/nd0821-c3-starter-code/starter/model/lb.pkl', 'rb'))
    encoder = pickle.load(open('/home/purusanth/project3/nd0821-c3-starter-code/starter/model/encoder.pkl', 'rb'))
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

    X_test, y_test, encoder, lb = process_data(
    data, categorical_features=cat_features, label=None, training=False,encoder=encoder,lb=lb
    )
    prediction= model.predict(X_test)[0]
    print(prediction)
    return {"prediction":str(prediction)}
