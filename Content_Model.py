from helper_functions import *
import Dataset
import warnings;

warnings.simplefilter('ignore')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


class ContentModel:
    def __init__(self):
        # self.dataset = Dataset.Dataset()
        self.model = TfidfVectorizer(min_df=0.0)
        self.model_features = None

    def fit(self, dataset=None):
        if dataset is None:
            print("Must pass a Pandas Dataframe to perform the prediction")
            raise ValueError("Must pass a Pandas Dataframe to perform the prediction")
        # Get the features of title text. The word is he feature here.
        self.model_features = self.model.fit_transform(dataset['title'])

    def predict(self, dataset, product_number=None, num_results=5):

        if self.model_features is None:
            print("Must fit the model before the prediction")
            raise ValueError("Must fit the model before the prediction")

        if product_number is None:
            print("Must specify a product to make the prediction")
            raise ValueError("Must specify a product to make the prediction")

        # print(dataset['asin'].head(10))
        product_ID = dataset.loc[dataset['asin'] == product_number].index[0]
        # print(product_ID)

        # doc_id: product id in given corpus
        # pairwise_dist will store the distance from given input apparel to all remaining apparels
        pairwise_dist = pairwise_distances(
            self.model_features, self.model_features[product_ID])

        # np.argsort will return indices of smallest distances
        indices = np.argsort(pairwise_dist.flatten())[0:num_results]
        # pdists will store the smallest distances
        pdists = np.sort(pairwise_dist.flatten())[0:num_results]

        predictions = []
        for i in range(0, len(indices)):
            # we will pass 1. doc_id, 2. title1, 3. title2, url, model

            get_result(indices[i], dataset['title'].iloc[indices[0]], dataset['title'].iloc[indices[i]],
                       dataset['imageURLHighRes'].iloc[indices[i]], self.model, self.model_features)
            print('ASIN :', dataset['asin'].iloc[indices[i]])
            print('BRAND :', dataset['brand'].iloc[indices[i]])
            print('Title:', dataset['title'].iloc[indices[i]])
            print('Eucliden distance from the given image :', pdists[i])
            print('=' * 125)

            predictions.append(dataset['asin'].iloc[indices[i]])

        return predictions

# tfidf_model(26594, 10)
