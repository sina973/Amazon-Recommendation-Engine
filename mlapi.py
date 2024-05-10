# Bring in lightweight dependencies
import json

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import Dataset
import pickle

app = FastAPI()


class ScoringItem(BaseModel):
    ProductID: str
    number_of_results: int

def create_dataset():
    meta_data = Dataset.Dataset()
    meta_data.create_meta_dataset(download=True)
    return meta_data

with open('CollaborativeModel.pkl', 'rb') as f:
    colab_model = pickle.load(f)

with open('ContentModel.pkl', 'rb') as f:
    content_model = pickle.load(f)


data = create_dataset()


@app.post('/')
async def recommend(item: ScoringItem):
    dictionary = item.dict()
    print('=' * 100)
    print(dictionary['ProductID'])
    print(dictionary['number_of_results'])
    print('=' * 100)
    colab_prediction = colab_model.predict(dictionary['ProductID'], dictionary['number_of_results'])
    content_prediction = content_model.predict(data.dataset, dictionary['ProductID'], dictionary['number_of_results'])

    print(content_prediction)

    return {"colaboration model prediction": colab_prediction, "content model prediction": content_prediction}