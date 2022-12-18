#!/usr/bin/env python
# coding: utf-8

# #### Create a machine learning model to detect fraudulent transactions.Fraud detection is an important application of machine learning in the financial services sector. This solution will help Xente provide improved and safer service to its customers.

# #  Exploratory Data Analysis

# In[1]:


# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# uplaudding and reading the data
variables = pd.read_csv("Xente_Variable_Definitions.csv")
variables


# In[3]:


train_dataset = pd.read_csv("training.csv")
train_dataset


# In[4]:


test_dataset = pd.read_csv("test.csv")
test_dataset


# In[5]:


# check the overview of the data

train_dataset.info()


# In[ ]:





# In[6]:


train_dataset.isnull().sum()


# In[7]:


test_dataset.info()


# In[8]:


test_dataset.isnull().sum()


# In[9]:


#checking the shape of the train and test datasets
tr = train_dataset.shape
te = test_dataset.shape
print("train_set_shape is: {} and test_set_shape is: {}".format(tr,te))
     


# In[10]:


#removing duplicate data if any
train_dataset.drop_duplicates(keep="first", inplace=True) 
test_dataset.drop_duplicates(keep="first", inplace=True)


# In[11]:


# we want to take a look at the fraudResult 
# distribution of legit transaction and fraud transaction

print("legit: {}".format((train_dataset["FraudResult"]==0).sum()))
print("Fraudulent: {}".format((train_dataset["FraudResult"]==1).sum()))


# In[12]:


# visulaizing the fradulent and non fradulent transaction
plt.bar("Fraudulent", train_dataset["FraudResult"].value_counts()[1], color="red")
plt.bar("legit", train_dataset["FraudResult"].value_counts()[0], width=0.5, color="green")
plt.ylabel("Count", fontsize=14)
plt.title("Fraudulent VS legit")


# In[13]:


# separating the two transaction
legit_transaction = train_dataset[train_dataset.FraudResult == 0]  
fraud_transaction = train_dataset[train_dataset.FraudResult == 1]
print(legit_transaction.shape)
print(fraud_transaction.shape)


# In[14]:


# statistical measures of the legit and fraud data
legit_transaction.Value.describe()


# In[15]:


fraud_transaction.Value.describe()


# ###### Comment
# the mean of the legit transaction is about 66,763 USD, 25% of the transaction value are less than 250 USD dollar, 50% of the legit  transaction are less than 1000 USD and 75% of the transaction is less than 5000 USD
#  
# for the fraud activities, the mean is about 1.6m USD, 25% of the fraud transaction is less than 500,000 USD, 50% of the fraud activities is  650, 000 USD and 75% is less than 2m USD.
# 
# hence even though there are more activities in the legit transaction, there it can be seen that fraud transaction has high monetary activities

# In[16]:


# compare the values for both transaction
train_dataset.groupby("FraudResult").mean()


# In[17]:


#legit_transaction.Amount.describe()


# In[18]:


#fraud_transaction.Amount.describe()


# In[19]:


# check for unique values in the train data (cardinality)
for var in train_dataset:
    print(var, "contains", len(train_dataset[var].unique()), "unique values")


# In[20]:


# check for unique values in the test data (cardinality)
for var in test_dataset:
    print(var, "contains", len(train_dataset[var].unique()), "unique values")


# In[21]:


# comparing the two dataset unique values
unique_values_dict = []
for var in test_dataset.columns:
    num_unique_train = len(train_dataset[var].unique())
    num_unique_test = len(test_dataset[var].unique())
    unique_values_dict.append({"Total_uinique_train": num_unique_train,
                              "Total_unique_test": num_unique_test})
    
    
df_unique = pd.DataFrame(unique_values_dict, index = test_dataset.columns)
df_unique


# In[22]:


train_dataset.ProductCategory.value_counts()


# In[23]:


train_dataset.CustomerId.value_counts()


# #### comment
# 
# - **TransactionStartTime** in this column, we will extract the datetime and see them in a visualization to know if they useful after extraction in a later column
# - The **transactionId**  as this has a large number attached to each transaction, cause it is a unique ID attached to each transaction
# - The **CurrencyCode, CountryCode** will be dropped as well since we are dealing with one country and is of no use here.
# - every other column, i will check them later with a visualization and correlation in addtion to this to determine eithere to use them or drop them

# In[24]:


columns = train_dataset.columns.tolist()[1:11]
columns


# 
# 

# ## Feature Engineering

# In[25]:


# creating and debit or credit features
def debit_or_credit(value):
    if value < 0:
        return "Credit"
    if value > 0:
        return "Debit"
train_dataset["Transaction"] = train_dataset["Amount"].apply(debit_or_credit)
test_dataset["Transaction"] = test_dataset["Amount"].apply(debit_or_credit)
train_dataset


# In[26]:


test_dataset


# In[27]:


# changing the channelId to a numerical value since getting dummies 
# for this gives an error due to the differences in the train and test 
# dataset
from sklearn.preprocessing import LabelEncoder
train_columns= train_dataset[["AccountId", "SubscriptionId",
                         "CustomerId", "ProviderId",
                         "ProductId", "ProductCategory",
                        "ChannelId", "PricingStrategy", "BatchId"]]
test_columns = test_dataset[["AccountId", "SubscriptionId",
                         "CustomerId", "ProviderId",
                         "ProductId", "ProductCategory",
                        "ChannelId", "PricingStrategy", "BatchId"]]

le = LabelEncoder()
for col in train_columns:
    train_dataset[col] = le.fit_transform(train_dataset[col])

for col in train_columns:
    test_dataset[col] = le.fit_transform(test_dataset[col])

test_dataset


# In[28]:


# now we want to take a look at the TransactionStartTime and extract 
# the day, month and year
#train_dataset["TransactionStartTime"]

train_dataset["minute"] = pd.to_datetime(train_dataset.TransactionStartTime).dt.minute
train_dataset["hour"] = pd.to_datetime(train_dataset.TransactionStartTime).dt.hour
train_dataset["day"] = pd.to_datetime(train_dataset.TransactionStartTime).dt.dayofweek
train_dataset["month"] = pd.to_datetime(train_dataset.TransactionStartTime).dt.month
train_dataset


# In[29]:


# doing same for the test dataset

test_dataset["hour"] = pd.to_datetime(test_dataset.TransactionStartTime).dt.hour
test_dataset["minute"] = pd.to_datetime(test_dataset.TransactionStartTime).dt.minute
test_dataset["day"] = pd.to_datetime(test_dataset.TransactionStartTime).dt.dayofweek
test_dataset["month"] = pd.to_datetime(test_dataset.TransactionStartTime).dt.month   
test_dataset    


# ### comment
# the month, day of the week, the minute and hour can have effect on the fraud actiivites, so we needed it and hence reason for extraction it

# In[30]:


correlations = train_dataset.corr()
fig = plt.figure(figsize = (16, 10))

sns.heatmap(correlations, annot = True, fmt = ".2f", linewidths = 0.8, cmap = "mako", square = True)
plt.show()


# In[31]:


# Visualizing correlations of the various features to fraud_result
fig = plt.figure(figsize = (16, 10))
(correlations
     .FraudResult
     .drop("FraudResult") # can't compare the variable under study to itself
     .sort_values(ascending=False)
     .plot
     .barh(figsize=(9,7)))
plt.title("correlation bar_hist")


# ### comment
# - **Amount** and value are a very imortant part for this detection, as can be seen from the heat map and visualization correlation. **Value** seems to have the higest correlations as well and as **Value** is the the absolute of the **Amount**,  we drop the **Amount** and use **Value**, but we still extract the signs as negative and positive as done earlier already to use as a feature 
# 
# 
# #### comments
# 
# 
# 
# ** ProviderId, PricingStrategy, ProductId, SuBscriptionI, BatchId, accounId, SubscriptionId, and TransactionStartTime** will all be droped, this a because they have a large amount of unique values as checked early accompanied by their negative correlation as seen here and so, they are not useful here.
# 
# 
# 
# - **ProductCategory** ProductIds are organized into these broader product categories. Which is okay and good to use
# - **ChannelId** Identifies if customer used web,Android, IOS, pay later or checkout. even though this has different unique values for the train and test data, it is important in this case, so will be used
# 
# - **Month, Day, and Minute** have a negative correlation on the Fraudresult and so it will not be used
# 
# - **Hour** even though the correlation is amll, we we will use the hour here since it has a positive correlation
# 

# In[32]:


# # get the categorical data in our train data
# categorical = [var for var in train_dataset.columns if train_dataset[var].dtypes == "O"]
# categorical


# In[33]:


# # for numerical data
# numerical =  [var for var in train_dataset.columns if train_dataset[var].dtypes != "O"]
# numerical


# In[34]:


# select a new dataset to use and get dummies for it
Transaction_dummies = pd.get_dummies(train_dataset.Transaction)
prodcat_dummies = pd.get_dummies(train_dataset.ProductCategory)
prodcat_dummies.columns = ["financial_services", "airtime ", "utility_bill", "data_bundles ", "tv", "ticket", "movies", "transport", "other"]
prodcat_dummies

train_dataset = pd.concat([train_dataset, Transaction_dummies, prodcat_dummies ], axis = 1)


# In[35]:


# # get the categorical data in our train data
# categorical = [var for var in train_dataset.columns if train_dataset[var].dtypes == "O"]
# categorical


# In[36]:


# train_dataset


# In[37]:


# # Standardization 
# # we will standardize because of the customerId and value
# # i will not be standardizing the value since it is an important detector in the fraudulent
# # activities, i remove it and add it later
# va_add = train_dataset["Value"]
# va_add =pd.DataFrame(va_add.values)
# va_add.columns = ["Value"]
# va_add


# In[38]:


# train_dataset.drop(["Value", "TransactionId", "CurrencyCode", "TransactionStartTime"], axis = 1, inplace = True)


# In[39]:


# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# train_dataset = scaler.fit_transform(train_dataset)


# ## Preprocessing

# In[ ]:





# In[40]:


legit_transaction = train_dataset[train_dataset.FraudResult == 0]  
fraud_transaction = train_dataset[train_dataset.FraudResult == 1]
print(legit_transaction.shape)
print(fraud_transaction.shape)


# In[41]:


legit_sample = legit_transaction.sample(n=193)
legit_sample


# In[42]:


train_dataset = pd.concat( [fraud_transaction, legit_sample] )
train_dataset


# In[43]:


train_dataset.columns


# In[44]:


# getting our x and y features
X = train_dataset[["ChannelId", "Credit", "Debit", "financial_services", "airtime ","data_bundles ", "utility_bill", "tv", "ticket", "movies", "transport", "other", "Value", "hour"]]
y = train_dataset["FraudResult"]


# In[45]:


# split the data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[46]:


X_train.shape, X_test.shape


# In[47]:


X_train.dtypes


# In[49]:


# now we train our model
from sklearn.linear_model import LogisticRegression
df_model = LogisticRegression(solver ="liblinear", random_state = 0)
df_model.fit(X_train, y_train)


# In[50]:


y_pred_test = df_model.predict(X_test)
y_pred_test


# In[51]:


# proabilty of legit transation
df_model.predict_proba(X_test)[:,0]


# In[52]:


# proabilty of fraud
df_model.predict_proba(X_test)[:,1]


# In[53]:


# check the accuracy of the model

from sklearn.metrics import accuracy_score

print(" model accuracy(test) is: ", accuracy_score(y_test, y_pred_test))


# In[55]:


# let get the accuracy for train just to make sure we are 
# not overfitting
y_pred_train = df_model.predict(X_train)
print("model accuracy(train) is :", accuracy_score(y_train, y_pred_train))


# In[57]:


# model accuracy metrics for logistic regression
print("training set score : ", df_model.score(X_train, y_train))
print("test set score : ", df_model.score(X_test, y_test))


# In[58]:


y_test.value_counts()


# In[59]:


null_accuracy = 23876/(len(y_test)) 
null_accuracy


# In[60]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
cm


# In[61]:


from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = df_model.classes_)
disp.plot()


# In[62]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))


# the data is imbalanced, so i used the data as it is in its imabalance form, but i had a model that overfitted and is only recognizing one side and so as a result i decided to do an under sampling and retraining the model.
# for the first model before retrainin, i had an accuracy score of 0.9984 for both the train and test data, null accuracy of very close range to the score accuracy. the recall, F1 scor and Precision are all 1 for the non_fraudulent transaction, but for the fraudulent transaction, it is 0.55 for precision and only 0.39 for the F1 score. 
# 
# as a result, i switched for sampling made the siyuation worst, like the model is accuracy is poor even though there Was a good null_accuracy, 
# 
# so i wpuld still stick the first one without sampling whch gave a better result than the second attempt. but the best solution is trying other models!
