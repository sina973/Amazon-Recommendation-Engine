import Dataset
import Content_Model
import Collaborative_Model
from helper_functions import *
import pickle

def save_model(model, model_name):
    file_name = model_name + ".pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_name):
    file_name = model_name + ".pkl"
    # Load the model from the file
    with open(file_name, 'rb') as f:
        loaded_model = pickle.load(f)

    return loaded_model

if __name__ == '__main__':

    meta_data = Dataset.Dataset()
    meta_data.create_meta_dataset(download=True)

    contentModel = Content_Model.ContentModel()
    contentModel.fit(meta_data.dataset)
    save_model(contentModel, "ContentModel")
    # contentModel = load_model("ContentModel") 26598 5120053475

    contentModel.predict(dataset=meta_data.dataset, product_number="B00063W3EW", num_results=5)

    review_data = Dataset.Dataset()
    review_data.create_review_dataset(download=True)

    collabModel = Collaborative_Model.CollaborativeModel()
    collabModel.fit(review_data)
    save_model(collabModel, "CollaborativeModel")
    # collabModel = load_model("CollaborativeModel")

    collabModel.predict(product_number="B00063W3EW", num_results=10)
