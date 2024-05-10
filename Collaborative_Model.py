from helper_functions import *
import Dataset
import warnings; warnings.simplefilter('ignore')
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


class CollaborativeModel:
    def __init__(self):
        # self.dataset = Dataset.Dataset()
        self.model = TruncatedSVD(n_components=10)
        self.decomposed_matrix = None
        self.ratings_matrix = None


    def fit(self, dataset):
        # Fit the Model
        self.ratings_matrix = dataset.get_pivot_table()
        self.decomposed_matrix = self.model.fit_transform(self.ratings_matrix)

    def predict(self, product_number=None, num_results=5):

        if self.decomposed_matrix is None:
            print("Must fit the model before the prediction")
            raise ValueError("Must fit the model before the prediction")

        if product_number is None:
            print("Must specify a product to make the prediction")
            raise ValueError("Must specify a product to make the prediction")


        # Correlation Matrix
        correlation_matrix = np.corrcoef(self.decomposed_matrix)

        # Index # of product ID purchased by customer
        # i = "5120053475"

        product_names = list(self.ratings_matrix.index)

        product_ID = product_names.index(product_number)

        correlation_product_ID = correlation_matrix[product_ID]

        recommendation = list(self.ratings_matrix.index[correlation_product_ID > 0.65])

        # Removes the item already bought by the customer
        recommendation.remove(product_number)

        result = recommendation[:num_results]

        print(result)

        return result

