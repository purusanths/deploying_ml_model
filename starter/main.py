from fastapi import FastAPI, Request
import pickle


app = FastAPI()

@app.get("/") ## define an endpoint
async def greeting():
    return {"Greetings": "Welcome to the MLops World!"}


@app.post("/inference")
async def inference(request: Request):

    data = await request.json()
    data = pd.Dataframe([data])

    model = pickle.load(open('./model/rfc_model.pkl', 'rb'))

    return model.predict(data)