from turtle import st
from fastapi import FastAPI, Request
import pickle
from pydantic import BaseModel,Field
from starter.ml.data import process_data
import pickle
import pandas as pd

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Welcoem to the MLops worlds"}



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

    model = pickle.load(open('/starter/model/rfc_model.pkl', 'rb')) #/home/purusanth/project3/nd0821-c3-starter-code
    lb = pickle.load(open('/starter/model/lb.pkl', 'rb'))# /home/purusanth/project3/nd0821-c3-starter-code
    encoder = pickle.load(open('/starter/model/encoder.pkl', 'rb')) #/home/purusanth/project3/nd0821-c3-starter-code
    cat_features = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

    X_test, y_test, encoder, lb = process_data(
    data, categorical_features=cat_features, label=None, training=False,encoder=encoder,lb=lb
    )
    prediction= model.predict(X_test)[0]
    print(prediction)
    return {"prediction":str(prediction)}
