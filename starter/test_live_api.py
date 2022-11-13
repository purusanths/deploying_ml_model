import requests
import json

data = json.dumps({
        "age": 45,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
        })
response = requests.post("http://127.0.0.1:8000/inference/",data=data)

print(response.status_code)
print(response.json())