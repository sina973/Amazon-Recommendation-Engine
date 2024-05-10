import pandas as pd
import gzip
import requests
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import nltk
from nltk.corpus import stopwords


def merge_datasets(dataset1, dataset2):
    return pd.concat([dataset1, dataset2])


class Dataset(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = pd.DataFrame()

    def read_gzip_json(self, file):
        g = gzip.open(file, 'rb')
        for line in g:
            yield json.loads(line)
        g.close()

    def read_data(self, file, num_datapoints):
        index = 0
        df = {}
        for d in self.read_gzip_json(file):
            df[index] = d
            index += 1
            if index == num_datapoints:
                break
        return pd.DataFrame.from_dict(df, orient='index')

    def text_processing(self, stop_words, text_, index, column, dataset):
        if type(text_) is not int:
            # print(text_)
            string = ""
            for words in text_.split():
                # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
                word = ("".join(e for e in words if e.isalnum()))
                # Conver all letters to lower-case
                word = word.lower()
                # stop-word removal
                if not word in stop_words:
                    string += word + " "
            dataset[column][index] = string

    def create_dataset(self, download=False, url=None, file_name=None, num_datapoints=100000):

        if download:
            if url is None:
                print("No URL found")
                raise ValueError("No URL found")

            else:
                if file_name is None:
                    print("No file name found")
                    raise ValueError("No file name found")

                print(f"Downloading {file_name} dataset...")

                # Send an HTTP GET request to download the dataset
                response = requests.get(url, verify=False)

                # Check if the request was successful
                if response.status_code == 200:
                    # Save the content of the response to a file
                    with open(file_name, "wb") as f:
                        f.write(response.content)
                    print("Dataset downloaded successfully.")
                else:
                    print("Failed to download dataset. Status code:", response.status_code)

                self.dataset = self.read_data(file_name, num_datapoints)

                return self.dataset

        else:
            if file_name is None:
                print("No file name found")
                raise ValueError("No file name found")

            print(f"Loading {file_name} dataset...")

            self.dataset = self.read_data(file_name, num_datapoints)

            print("Dataset loaded successfully.")

            return self.dataset

    def create_review_dataset(self, download=True, num_datapoints=250000):

        url1 = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Electronics.json.gz"
        file_name1 = "Electronics.json.gz"

        # URL of the dataset to download
        url2 = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz"
        file_name2 = "Clothing_Shoes_and_Jewelry.json.gz"

        if download:
            df1 = self.create_dataset(download=True, url=url1, file_name=file_name1, num_datapoints=num_datapoints)
            df2 = self.create_dataset(download=True, url=url2, file_name=file_name2, num_datapoints=num_datapoints)
        else:
            df1 = self.create_dataset(download=False, url=None, file_name=file_name1, num_datapoints=num_datapoints)
            df2 = self.create_dataset(download=False, url=None, file_name=file_name2, num_datapoints=num_datapoints)

        self.dataset = merge_datasets(df1, df2)

        self.review_data_cleaning(self.get_unique_products())

        return self.dataset

    def create_meta_dataset(self, download=True, num_datapoints=100000):

        meta_url1 = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Electronics.json.gz"
        file_name1 = "meta_Electronics.jsonl.gz"

        meta_url2 = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"
        file_name2 = "meta_Clothing_Shoes_and_Jewelry.jsonl.gz"

        if download:
            df1 = self.create_dataset(download=True, url=meta_url1, file_name=file_name1, num_datapoints=num_datapoints)
            df2 = self.create_dataset(download=True, url=meta_url2, file_name=file_name2, num_datapoints=num_datapoints)
        else:
            df1 = self.create_dataset(download=False, url=None, file_name=file_name1, num_datapoints=num_datapoints)
            df2 = self.create_dataset(download=False, url=None, file_name=file_name2, num_datapoints=num_datapoints)

        self.dataset = merge_datasets(df1, df2)
        # self.dataset = pd.concat([df1, df2])

        # print(dataset.head())

        self.save_unique_products()

        self.meta_data_cleaning()

        return self.dataset

    def save_unique_products(self):
        unique_products = self.dataset['asin'].unique()
        with open('unique_products.txt', 'w') as f:
            for product in unique_products:
                f.write(product + '\n')

    def get_unique_products(self):
        with open('unique_products.txt', 'r') as f:
            unique_products = f.read().split('\n')
            return unique_products

    def process_columns(self, type=None):
        if type is None:
            print("Must enter the type of dataset. Either 'meta' or 'review'.")
            raise ValueError("There is no type for the dataset.")

        if type == 'meta':
            columns = ['asin', 'brand', 'imageURLHighRes', 'category', 'title']
        else:
            columns = ['asin', 'reviewerID', 'overall', 'reviewText', 'summary']

        self.dataset = self.dataset[columns]

        return self.dataset

    def remove_list(self):
        self.dataset = self.dataset.applymap(lambda x: x if not isinstance(x, list) else x[0] if len(x) else np.nan)
        return self.dataset

    def remove_nan(self):
        self.dataset.dropna(inplace=True)
        return self.dataset

    def sort_and_remove_small_titles(self):
        data_sorted = self.dataset[self.dataset['title'].apply(lambda x: len(x.split()) > 4)]
        data_sorted.sort_values('title', inplace=True, ascending=False)

        self.dataset = data_sorted
        return self.dataset

    def visualize_data_types(self):
        abs_values = self.dataset['category'].value_counts(ascending=False)
        rel_values = self.dataset['category'].value_counts(
            ascending=False, normalize=True).values * 100

        plt.subplots_adjust(top=1.1)

        ax = sns.countplot(data=self.dataset, x="category")
        ax.bar_label(container=ax.containers[0], labels=abs_values)
        plt.show()

    def clean_titles(self):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))  # English stop words.
        # print(self.dataset.head())
        for index, row in self.dataset.iterrows():
            # print(row['title'])
            # print(row['title'].split())
            self.text_processing(stop_words, row['title'], index, 'title', self.dataset)

    def save_dataset(self, file_name='Amazon_sorted_data.csv'):
        self.dataset.to_csv(file_name, index=False)

    def clean_dataset(self, unique_products):
        self.dataset = self.dataset[self.dataset['asin'].isin(unique_products)]
        return self.dataset

    def calc_popular_products(self):
        popular_products = pd.DataFrame(self.dataset.groupby('asin')['overall'].count())
        return popular_products.sort_values('overall', ascending=False)

    def rating_per_user(self):
        no_of_rated_products_per_user = self.dataset.groupby(by='reviewerID')['overall'].count().sort_values(
            ascending=False)

        no_of_rated_products_per_user.head()

        quantiles = no_of_rated_products_per_user.quantile(np.arange(0, 1.01, 0.01), interpolation='higher')

        plt.figure(figsize=(10, 10))
        plt.title("Quantiles and their Values")
        quantiles.plot()
        # quantiles with 0.05 difference
        plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
        # quantiles with 0.25 difference
        plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label="quantiles with 0.25 intervals")
        plt.ylabel('No of ratings by user')
        plt.xlabel('Value at the quantile')
        plt.legend(loc='best')
        plt.show()

    def visualize_rating_per_product(self):
        no_of_ratings_per_product = self.dataset.groupby(by='asin')['overall'].count().sort_values(
            ascending=False)

        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = plt.gca()
        plt.plot(no_of_ratings_per_product.values)
        plt.title('# RATINGS per Product')
        plt.xlabel('Product')
        plt.ylabel('No of ratings per product')
        ax.set_xticklabels([])

        plt.show()

    def visualize_ratings(self):
        self.dataset.groupby('asin')['overall'].mean().head()

        self.dataset.groupby('asin')['overall'].mean().sort_values(ascending=False).head()

        # Total no of rating for product

        self.dataset.groupby('asin')['overall'].count().sort_values(ascending=False).head()

        ratings_mean_count = pd.DataFrame(self.dataset.groupby('asin')['overall'].mean())

        ratings_mean_count['rating_counts'] = pd.DataFrame(self.dataset.groupby('asin')['overall'].count())

        ratings_mean_count.head()

        ratings_mean_count['rating_counts'].max()

        plt.figure(figsize=(8, 6))
        plt.title('# rating')
        plt.xlabel('# rating')
        plt.ylabel('count')
        plt.rcParams['patch.force_edgecolor'] = True
        ratings_mean_count['rating_counts'].hist(bins=50)

        plt.figure(figsize=(8, 6))
        plt.title('rating')
        plt.xlabel('rating')
        plt.ylabel('Count')
        plt.rcParams['patch.force_edgecolor'] = True
        ratings_mean_count['overall'].hist(bins=50)

        plt.figure(figsize=(8, 6))
        plt.rcParams['patch.force_edgecolor'] = True
        sns.jointplot(x='overall', y='rating counts', data=ratings_mean_count, alpha=0.4)

    def visualize_popular_products(self):
        most_popular = self.calc_popular_products()
        most_popular.head(30).plot(kind="bar")

    def get_pivot_table(self):
        ratings_matrix = self.dataset.pivot_table(values='overall', index='reviewerID', columns='asin',
                                                  fill_value=0)
        return ratings_matrix.T

    def meta_data_cleaning(self):
        self.dataset = self.process_columns(type='meta')
        self.dataset = self.remove_list()
        self.dataset = self.remove_nan()
        self.dataset.drop_duplicates(inplace=True)
        self.dataset['title'].drop_duplicates(inplace=True)
        self.dataset = self.sort_and_remove_small_titles()
        self.clean_titles()

    def review_data_cleaning(self, unique_products=None):
        if unique_products is None:
            print("Must specify products to filter")
            raise ValueError("Must specify products to filter")

        self.dataset = self.clean_dataset(unique_products)
        self.dataset = self.process_columns(type='review')
