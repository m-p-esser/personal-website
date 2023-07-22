---
author: Marc-Philipp Esser
title: Dealing with categorical data
date: 2019-12-29
description: Dealing with categorical data
tags: 
- encoding
- feature-engineering
- data-science
- python
keywords: 
- encoding
- feature-engineering
- data-science
- python
---

# Introduction

In real-world datasets it is often the case that you have a mixed variable types. While **Machine learning algorithmns** usually can handle only numerical data, they **can't work with categorical variables**. Being able to use categorical variables in modeling makes it necessary to transform those variables. This process is called **Encoding** and there are many different strategies for it.

There are two types of variables which can be seen as categorical variables:
1. **Nominal variables**: Nominal data is made of discrete values with no numerical relationship between the different values — mean and median are meaningless. An example here would be the colour of a car or the job title of a person.

2. **Ordinal variables**: A variable used to rank a sample of individuals with respect to some characteristics, but differences (i.e., intervals) and different points of the scale are not necessarily equivalent. An example here would be the 

<br>

# Basic Encoding Strategies

The following *encoding strategies* are easy to understand and very popular in Machine Learning:

- One Hot Encoding
- Ordinal Encoding
- Binary Encoding
- Frequency Encoding
- Hashing Encoding
- Sum Encoding
- Mean Encoding
- Leave One Out Encoding

To illustrate these different encoding strategies i will be using an *sample dataset* which i manuelly created in Excel. You can find it in this [Github Repository](https://github.com/m-p-esser/dealing-with-categorical-data).

Okay let's begin by importing all necessary python packages. If your working with the [Anaconda Distribution](https://www.anaconda.com/distribution/) all packages except for the ``category_encoders`` module should be preinstalled. Installation instructions for this package can be found [here](https://contrib.scikit-learn.org/categorical-encoding/).


```python
# import packages
import os
import numpy as np
import pandas as pd
import category_encoders as ce

# load dataset
root_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(root_dir, 'data')
df = pd.read_excel(os.path.join(data_dir, 'dummy_dataset.xlsx'))

# seperate predictor and target variable
X = df.loc[:, ['id', 'age', 'iq', 'hair']]
y = df.loc[:, ['target']]
```

Let's have a look at the dataset:


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>iq</th>
      <th>hair</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0-17</td>
      <td>very low</td>
      <td>brown</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>18-29</td>
      <td>low</td>
      <td>black</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>30-49</td>
      <td>medium</td>
      <td>blonde</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>50-69</td>
      <td>high</td>
      <td>black</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>70+</td>
      <td>very high</td>
      <td>blonde</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>18-29</td>
      <td>medium</td>
      <td>brown</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>30-49</td>
      <td>high</td>
      <td>blonde</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>30-49</td>
      <td>low</td>
      <td>brown</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>50-69</td>
      <td>medium</td>
      <td>brown</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>18-29</td>
      <td>high</td>
      <td>black</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<br>

## One Hot Encoding

I'll mention this technique first since it is the go-to approach to encode data and very easy to understand. This approach is also called *dummy or indicator encoding*. When transforming a variable via One Hot Encoding <u>every unique value of the original columns get it's own column</u>. This means the number of columns in a dataframe gets increased by k-1 (where k is the number of unique values). This type of encoding can be applied to *nominal* as well as *categorical variables*.


```python
one_hot_enc = ce.OneHotEncoder(cols=['hair'], use_cat_names=True)
encoding = one_hot_enc.fit_transform(X['hair'], y)
combined = pd.concat([X['hair'], encoding], axis=1)
combined.sort_values('hair')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>hair_brown</th>
      <th>hair_black</th>
      <th>hair_blonde</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As you can see each hair color gets it's own column. Since a person can only have one hair color there is only a <u>single 1 in each row</u>. As you might also imagine the <u>number of columns created by this approach can very large</u> and therefore computionally and memory expensive especially if we deal with **high cardinality features**. High cardinality features are variables which have a lot of unique values. An good example here are zip codes.

<br>

## Ordinal Encoding

This technique as the name suggests can <u>only be applied to ordinal features</u>. Here each string will be replaced by an corresponding integer. This replacement make sense and is recommended since there is an *natural order* in ordinal variables. Furthermore this approach is **very cost effective** because no additional columns are created.


```python
ordinal_enc = ce.OrdinalEncoder(cols=['iq'], 
                                mapping=[{'col':'iq', 'mapping':
                                          {'very low': 1, 'low': 2, 'medium': 3, 
                                           'high': 4, 'very high': 5}}])
encoding = ordinal_enc.fit_transform(X['iq'], y).rename(columns={'iq':'iq_enc'})
combined = pd.concat([X['iq'], encoding], axis=1)
combined.sort_values('iq_enc')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iq</th>
      <th>iq_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>very low</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>low</td>
      <td>2</td>
    </tr>
    <tr>
      <td>7</td>
      <td>low</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>medium</td>
      <td>3</td>
    </tr>
    <tr>
      <td>5</td>
      <td>medium</td>
      <td>3</td>
    </tr>
    <tr>
      <td>8</td>
      <td>medium</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>high</td>
      <td>4</td>
    </tr>
    <tr>
      <td>6</td>
      <td>high</td>
      <td>4</td>
    </tr>
    <tr>
      <td>9</td>
      <td>high</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>very high</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



The following things happend when applying the encoding:
- The value **'very low'** got replaced by a **1**, the value **low** got replaced by a **2** and so fourth
- The <u>natural order has been preserved</u> because of the mapping argument which the encoder has been given. Otherwise the assignment would have been random.

<br>

## Binary Encoding

This encoding method can be seen as <u>hybrid between One Hot and Hashing Encoders</u>. It creates fewer features as the One Hot Encoding approach while preserving a more unique character of values. It works very well with *high dimensional ordinal data* altough this combination is very rare.


```python
binary_enc = ce.BinaryEncoder(cols=['hair'])
encoding = binary_enc.fit_transform(X['hair'], y)
combined = pd.concat([X['hair'], encoding], axis=1)
combined.sort_values('hair')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>hair_0</th>
      <th>hair_1</th>
      <th>hair_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Altough it first might seem like this technique is exactly like One Hot Encoding, the appearances are deceptive. If you observe the third row (index = 2) you will notice that there are two '1s' in the whole row. This wouldn't be possible in a One Hot Encoding. This row tells us that hair color of the respondent is ``blond``. Blond is the third type of hair colour and three is represented by 011 in the **binary language**.

**Binary Encoding follow these steps**:
- The categories are encoded by an Ordinal Encoder if they aren’t already in numeric form
- Then those integers are converted into binary code, so for example 5 becomes 101 and 8 becomes 112.
- Then the digits from that binary string are split into separate columns. So if there are 4–7 values in an ordinal column then 3 new columns are created: one for the first bit, one for the second, and one for the third.
- Each observation is encoded across the columns in its binary form.

<br>

## Frequency Encoding

This encoding approach is **rather uncommon** and can be used for *nominal* as well as *ordinal* features. In this case the values get replaced by their <u>frequency in relation to the whole dataset</u>, hence the name. In this case we won't use the ``category_encoders`` package. Instead we'll use panda methods to encode the data.


```python
freq_enc = df.groupby('hair').size() / len(df)
df.loc[:, 'hair_enc'] = df['hair'].map(freq_enc)
df.loc[:, ['hair', 'hair_enc']].sort_values('hair').sort_values('hair_enc')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>hair_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>



Since the color brown is the most frequent value in the ``hair`` column is has the highest value in the encoded columns followed by the other two values. Instead of relative value <u>we could also use absolute frequencies</u> (counts).

<br>

## Hashing Encoding  

This technique implements the **hashing trick**. It is *similar to one-hot encoding* but with less newly created columns and some loss in information because of *collision effects*. The collisions do not significantly affect performance unless there is a great deal of overlap. An detailed discussion of this method can be found [here](https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087) and an in-depth explanation can be found in [this article](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f).


```python
hash_enc = ce.HashingEncoder(cols=['hair'])
encoding = hash_enc.fit_transform(X['hair'], y)
combined = pd.concat([X['hair'], encoding], axis=1)
combined.sort_values('hair')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
      <th>col_3</th>
      <th>col_4</th>
      <th>col_5</th>
      <th>col_6</th>
      <th>col_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Since the <u>number of dimensions defaults to 8</u>, the same amount of columns get created by the encoding. Since there are only three unique values in the ``hair`` column we see five columns with only zeros. **Hashing gets interesing** when we have a <u>lot of unique values</u> and the number of newly added columns should be smaller than the amount of unique values. In *Kaggle* competitions hashing has been a very sucessfull method for encoding high cardinality features. 

<br>

## Sum Encoding  

A **Sum Encoder** compares the mean of the dependent variable (``target``) for a given level of a categorical column to the overall mean of the target. This method is <u>very similar to One Hot Encoding</u> except that the number of created columns is always one less. This is the case because one <u>unique value is always held constant</u>. 


```python
sum_enc = ce.SumEncoder(cols=['age'])
encoding = sum_enc.fit_transform(X['age'], y)
combined = pd.concat([X['age'], encoding, y], axis=1)
combined.sort_values('age')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>intercept</th>
      <th>age_0</th>
      <th>age_1</th>
      <th>age_2</th>
      <th>age_3</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0-17</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>18-29</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>18-29</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>18-29</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>30-49</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>30-49</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>30-49</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>50-69</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>50-69</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>70+</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



That one value is always contant can be observed in row 5 (index = 4), where the value for *70+* is always encoded as *-1* regardless of the column. Sum Encoding is is commonly used in Linear Regression (LR) types of models.

<br>

## Target Encoding  

Target Encoding or Mean Encoding directly correlates the encoded variable with a discrete target variable. So this approach can <u>only be used in classification problems</u>. The danger of this method lies within the **problem of overfitting** which can only be addressed by *regularization*. Nevertheless this Encoding technique has been very sucessfully used in *Kaggle* competitions.


```python
target_enc = ce.TargetEncoder(cols=['hair'])
encoding = target_enc.fit_transform(X['hair'], y).rename(columns={'hair':'hair_enc'})
combined = pd.concat([X['hair'], encoding, y], axis=1)
combined.sort_values('hair')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>hair_enc</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0.6468</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0.6468</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>0.6468</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0.3532</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0.3532</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0.3532</td>
      <td>1</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>0.5000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>0.5000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>0.5000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>0.5000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As you can see there are only three different numeric values. Reason for this is that like in Frequence Encoding each unique category gets a new numerical value between 0 to 1. In the first row we can see that observations with the hair color ``brown`` are in 50% of the cases correlated to the target variable, hence the value 0.5.

<br>

## Leave One Out Encoding  

Leave-one-out Encoding (**LOO** or **LOOE**) is another example of a target-based encoder. The name speaks for itself: we compute the **mean target of category k** for observation i <u>if observation i would be removed</u> from the dataset.


```python
loo = ce.LeaveOneOutEncoder(cols=['hair'])
encoding = loo.fit_transform(X['hair'], y).rename(columns={'hair':'hair_enc'})
combined = pd.concat([X['hair'], encoding, y], axis=1)
combined.sort_values('hair')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hair</th>
      <th>hair_enc</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>black</td>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>black</td>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>black</td>
      <td>1.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>blonde</td>
      <td>0.500000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>blonde</td>
      <td>0.500000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>blonde</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>0</td>
      <td>brown</td>
      <td>0.666667</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>brown</td>
      <td>0.333333</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>brown</td>
      <td>0.333333</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>brown</td>
      <td>0.666667</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



There are other more complicated Target based Encoder like the *M-Estimate*, *Weight of Evidence* or *James-Steiner Encoder (WOE)* which also had their share of sucess in *Kaggle* competitions. If you're interested in those, check out this [blog post](https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8) on Towards Data Science.

# Summary

Here a quick summary of all presented encoding techniques in a tabular format. This should help you and me to maintain a clear overview:

| Technique              | What does it do?                                                                           | Variable Type    |
|------------------------|--------------------------------------------------------------------------------------------|------------------|
| One Hot Encoding       | Each category gets its own column                                                          | Nominal, Ordinal |
| Ordinal Encoding       | Each category gets mapped to an integer value                                              | Ordinal          |
| Binary Encoding        | Each category gets mapped to a binary code and columns split according to the code length. | Ordinal          |
| Frequency Encoding     | The frequency of each value in category gets calculated and mapped                         | Nominal, Ordinal |
| Hashing Encoding       | Replace values by hash code with varying length and columns split according to code length | Nominal, Ordinal |
| Sum Encoding           | Similar to One Hot Encoding, except one less column created                                | Nominal, Ordinal |
| Target Encoding        | Correlate variable to target variable                                                      | Nominal, Ordinal |
| Leave One Out Encoding | Similar to Target Encoding except current observation gets ignored in calculation          | Nominal, Ordinal |

<br>

# Final Discussion

Okay let's wrap this up by reflecting on the decision process when choosing an encoding method.

**Relevant questions** to ask when you think about which Encoding strategy to choose are:
- What <u>variable type</u> do i have? Ordinal or Nominal
- <u>How many unique values</u> does the variable have (cardinality)? Low, Medium or High 
- What type of <u>Machine Learning problem</u> do i try to solve? For example: Classification, Regression, Clustering

<br>

I can give you the following advise when choosing an technique to encode your data:

- First check the type of ML problem: Any kind of **Target Encoding** <u>only works for classification</u>
- In general the following Encodings make sense for **ordinal features**: <u>Ordinary, Binary, OneHot, Leave One Out, Target Encoding</u>
- If you have a **ordinal columns with a lot of features** (rare case) take a <u>Binary Encoder</u>. Leave One Out or Target Encoding also make sense.
- In general the following Encodings make sense for **nominal features**: <u>OneHot, Hashing, LeaveOneOut, and Target encoding </u>. Avoid OneHot for high cardinality columns

<br>

If you need a visual map to guide you through the decision, here is a flow chart i found:

[source](https://blog.featurelabs.com/encode-smarter/)
