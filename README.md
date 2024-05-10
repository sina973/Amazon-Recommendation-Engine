# Amazon-Recommendation-Engine
Two Amazon recommendation engines with API implementation

In this repository, I implemented two recommendation engines that work on Amazon Electronics and Clothing datasets.

# Models
In this project, I deployed a recommendation engine using two different ML models: collaborative filtering and content-based filtering. 
![visualization of the models](https://miro.medium.com/v2/resize:fit:720/format:webp/0*Ys_g_6dpXLf9pWH6)

## Content-Based Model
Content-based filtering is used to compare products in the dataset with each other and recommend the most similar products to what the user bought before. The similarity between products is based on their description, title, features, and other attributes of the products. You can find the implementation in [Content_Model.py](Content_Model.py) file. To implement this engine, I used the metadata of the Amazon dataset, which will be discussed in the Dataset section. < br / >

## Collaborative Filtering Model
Collaborative filtering, or Item-Item recommendation, is based on the idea that the best recommendations come from people who have similar tastes. In other words, it uses historical item ratings to predict how someone would rate an item. These methods are based on machine learning and data mining techniques. Compared to other approaches like the memory-based approach, these methods have the advantage of being able to recommend a greater number of items to a greater number of users. Even with huge, sparse matrices, they have a large coverage. I decided to use TruncatedSVD to deploy the model. You can find the implementation in [Collaborative_Model.py](Collaborative_Model.py) file. To implement this engine, I used the review from the Amazon dataset, which will be discussed in the Dataset section.


# Dataset
In this project, I used Amazon Electronics and Clothing datasets released in 2018. You can find the dataset via the link [Amazon Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/#files).

## Pre-processing of the data

### Meta Data
The preprocessing required for the metadata is based on the data frame itself. After downloading the two datasets and merging them together, a number of processing steps need to be applied for the data to be used for model prediction and evaluation.
1. There are many columns available in the dataset that will not be used for the purpose of building a recommendation engine. I decided to use only the productID, brand, title, category, and image URL columns of the data.
2. After removing redundant columns, there is a need to process the values of each column. I removed lists in columns and replaced them with the value inside those lists, removed NaN values, and removed duplicates.
3. There are many non-alphanumerical characters in the title column. I omitted those characters and removed stop words as well. Also, there are many titles which are just one or two words, therefore, I decided to remove small titles which are smaller than four words.
   
You can find the implementation in the [Dataset.py](Dataset.py) file.

### Review Data
The preprocessing steps here are much less than those for metadata.
1. There are many columns available in the dataset that will not be used for the purpose of building a recommendation engine just like the above. I decided to use only the productID, reviewerID, rating, review text, and summary of the product.
2. After removing redundant columns, there is a need to remove those products which are not in the meta data. I used the unique products in the metadata and filtered the products in the review data.
   
You can find the implementation in the [Dataset.py](Dataset.py) file.

## Visualization
There are many 




