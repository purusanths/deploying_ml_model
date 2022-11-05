from turtle import st
from fastapi import FastAPI, Request
import pickle
from pydantic import BaseModel,Field


class Taggeditem(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias='capital-gain')
    capital_loss: float =Field(alias='capital-loss')
    hours_per_week: float =Field(alias='hours-per-week')
    native_country: str =Field(alias='native-country')
    salary: float




app = FastAPI()

@app.get("/") ## define an endpoint
async def greeting():
    return {"Greetings": "Welcome to the MLops World!"}


@app.post("/inference")
async def inference(request: Taggeditem):

    data = await request.json()
    data = pd.Dataframe([data])

    model = pickle.load(open('./model/rfc_model.pkl', 'rb'))

    return model.predict(data)