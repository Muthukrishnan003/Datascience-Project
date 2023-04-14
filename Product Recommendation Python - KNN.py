#!/usr/bin/env python
# coding: utf-8

# # Product Recommendation Using Collaborative Filtering
# 
# ###             Build a recommendation system to recommend  the products to customers based on the their previous ratings for other products.

# # Load the Data Set
# 

# In[1]:


#importing the libraries thats we want
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Loading the dataset
# Here, I take the BigBasket Dataset (Online Grocery Store)

df= pd.read_csv('BigBasket Products.csv')


# In[3]:


# Printing the number of rows & columns in Bigbasket Dataset

rows_count, columns_count = df.shape
print('Total Number of rows :', rows_count)
print('Total Number of columns :', columns_count)


# In[4]:


# Display the Bigbasket dataset

df.head()


# # Data Cleaning
# 
# ### It involves filling of missing values, smoothing or removing noisy data and outliers along with resolving inconsistencies.

# In[5]:


# Information about the dataset

print (df.info())


# In[6]:


# Finding the null values Presented in the each column and Calculating the null values for each column

df.isnull().sum()


# In[7]:


# Here, we dropping the null value from the dataset

new = df.dropna()


# In[8]:


new.info()


# # Dimensionality Reduction
# 
# ### The number of input features, variables, or columns present in a given dataset is known as dimensionality, and the process to reduce these features is called dimensionality reduction.

# In[9]:


# Now, we removing the unwanted column.

var = ['index','sub_category', 'brand', 'type','description']
new = new.drop(var, axis='columns')


# In[10]:


# Displaying the dataset after removed some columns.

new.head()


# In[11]:


#Checking the dataset it has null value or not.
new.isnull().sum()


# In[12]:


# After cleaned dataset printing the no. of rows and columns.
rows_count, columns_count = new.shape
print('Total Number of rows :', rows_count)
print('Total Number of columns :', columns_count)


# In[13]:


# Now, Send the preprocessed dataset to create a new cleaned dataset.
# new.to_csv('Cleaned Dataset of Bigbasket.csv') 


# # Exploratory Data Analysis
# 
# ### Visualizing the Data Set to Choose the Model or Algorithm
# 

# In[14]:


#now i opening the cleaned dataset
cleaned = pd.read_csv("Cleaned Dataset of Bigbasket.csv")


# In[15]:


cleaned.head()


# In[16]:


#counting rating column 
cleaned['rating'].value_counts()


# ###  Here, I modifying the rating column to rate. Because I rounding the float value into a single digit value

# In[17]:


# Again,counting rating column.
cleaned['Rate'].value_counts()


# In[18]:


# Displaying the Rate and Total_counts in Dataframe
rating_counts = pd.DataFrame(cleaned['Rate'].value_counts()).reset_index()
rating_counts.columns = ['Rate', 'Total_Counts']
rating_counts


# In[19]:


#sns.countplot(x ='Rate',data=cleaned, hue='Rate')


# In[20]:


# Visualizing the Rate and Total_counts Using Bar Chart
a=rating_counts['Rate']
b=rating_counts['Total_Counts']
sns.barplot(x=a,y=b,orient='v')
plt.title("Vertical Bar Chart")
plt.show()


# ###    It's interesting that there are more people giving rating of  4 

# In[21]:


a=rating_counts['Rate']
b=rating_counts['Total_Counts']
sns.barplot(x=b,y=a,orient='h')
plt.title("Horizontal Bar Chart")
plt.show()


# In[40]:


import seaborn as sn
# Let us understand the distribution of each attributes (product_id, rating, rate)

# Distribution of Product
sn.displot(data=cleaned, x="product_id", kind="hist", aspect=1)

# Distribution of rating
sn.displot(data=cleaned, x="rating", kind="hist", aspect=1)

# Distribution of Rate
sn.displot(data=cleaned, x="Rate", kind="hist", aspect=1)


# In[23]:


cleaned.dtypes


# # Building the 1st Model - Implemeting the Nearest Neighbor Model

# ## KNN model for item-based collaborative filtering
# ###  Reshaping the Dataset
# ##### For K-Nearest Neighbors, we want the data to be in an array, where each row is a product and each column is a different user. To reshape the dataframe, we'll pivot the dataframe to the wide format with product as rows and users as columns. Then we'll fill the missing observations with 0s since we're going to be performing linear algebra operations (calculating distances between vectors). Finally, we transform the values of the dataframe into a scipy sparse matrix for more efficient calculations.

# In[24]:


#pivot table and create product-user matrix

product_mat = cleaned.pivot(index='product_id', columns='user_id', values='Rate').fillna(0)


# In[25]:


product_mat.head()


# In[26]:


product_mat.shape


# In[27]:


# create mapper from movie title to index

product_to_idx = {
    product: i for i, product in 
    enumerate(list(cleaned.set_index('product_id').loc[product_mat.index].product_name))
}


# In[28]:


# importing sprase matrix from scipy library
# convert dataframe of Product features to scipy sparse matrix
from scipy.sparse import csr_matrix  

product_sparse = csr_matrix(product_mat.values)


# In[29]:


product_sparse


# ### Fitting the Model
# 
# ##### Time to implement the model. We’ll initialize the NearestNeighbors class as model_knn and fit our sparse matrix to the instance. By specifying the metric = cosine, the model will measure similarity bectween artist vectors by using cosine similarity.
# 

# In[30]:


#importing knn algorithm from sklearn library
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(product_sparse)


# ### Here I use the fuzzywuzzy library to find the product name that is in dataset or not
# 

# In[31]:


#defining function
from fuzzywuzzy import fuzz

def fuzzy_matching(mapper, fav_product, verbose=True):
  
    match_tuple = []
    # get match
    for product_name, idx in mapper.items():
        ratio = fuzz.ratio(product_name.lower(), fav_product.lower())
        if ratio >= 60:
            match_tuple.append((product_name, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our dataset: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


# ## Here I write the function to recommend the product

# In[32]:


def make_recommendation(model_knn, data, mapper, fav_product, n_recommendations):
   
    # fit
    model_knn.fit(data)
    # get input product index
    print('You have input product:', fav_product)
    idx = fuzzy_matching(mapper, fav_product, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_product))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))


# ## Here I specifically entering the product that I want and It goes to make_recommendation method and then it gives the recommendation 

# In[33]:


# butter, rice, cookies, 
my_favorite = 'cookies'

make_recommendation(
    model_knn=model_knn,
    data=product_sparse,
    fav_product=my_favorite,
    mapper=product_to_idx,
    n_recommendations=4)


# ### Checking the Accuracy 

# In[34]:


# calcuate total number of entries in the product-user matrix
num_entries = product_mat.shape[0] * product_mat.shape[1]
# calculate total number of entries with zero values
num_zeros = (product_mat==0).sum(axis=1).sum()
# calculate ratio of number of zeros to number of entries
ratio_zeros = num_zeros / num_entries
print('There is about {:.2%} of ratings in our data is missing'.format(ratio_zeros))


# In[ ]:





# In[ ]:




