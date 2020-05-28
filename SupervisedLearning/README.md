# *COVID-19 Community Mobility Reports:​ Regression*

### Authors:
- Ana Mafalda Santos, up201706791
- Diogo Silva, up201706892
- João Luz, up201703782

## Introduction
---
The main purpose of this project was to build a regression model using different **Supervised Learning** algorithms for the given dataset: **CoV-19 Community Mobility Report**.

After analyzing and handling our dataset, we aim to predict the number of people infected by SARS-CoV-2 based on the variation of the population's mobility trends. To make it less biased, we'll predict the confirmed cases per million instead of using just the total amount of cases, as different countries have different population sizes.

## Description of the Problem/Dataset
---
Our dataset - **CoV-19 Community Mobility Report** - seeks to provide insights into what has changed due to policies aimed at combating COVID-19 and evaluate the changes in community activities and its relation to reduced confirmed cases of COVID-19. The reports chart movement trends, comparing to an expected baseline, over time by geography, across the following categories of places:

* **Retail & Recreation**: mobility trends for places like restaurants, cafes, shopping centers, theme parks, museums, libraries, and movie theaters.
* **Grocery & Pharmacy**: mobility trends for places like grocery markets, food warehouses, farmers' markets, specialty food shops, drug stores, and pharmacies.
* **Parks**: mobility trends for places like national parks, public beaches, marinas, dog parks, plazas, and public gardens.
* **Transit stations**: mobility trends for places like public transport hubs such as subway, bus, and train stations.
* **Workplaces**: mobility trends for places of work.
* **Residential**: mobility trends for places of residence.

The dataset also contains these CoV-19 statistics:

* **Total Cases**: Total number of people infected with the SARS-CoV-2.
* **Fatalities**: Total number of deaths caused by CoV-19.
* **Population**: Total number of inhabitants.

## Approach
---
We start by pre-processing the data we had. To compensate for missing and NaN values on the dataset, we used Median and Zero Imputation, to better infer those missing values from the existing part of the data. We also added new columns to extract more reliable information from the existing columns:

* **New Cases (percentage)**: variation of the total cases of CoV-19 cases in comparison to the previous day.
* **New Fatalities (percentage)**: variation of the total fatalities of CoV-19 cases in comparison to the previous day.
* **Cases per Million**: total cases CoV-19 cases per million people.

In order to improve the model, several ways of partitioning the input data into training and testing data have been considered, such as Train/Test Split and K-Fold Cross-Validation.

To predict the number of people infected based on the available data, we used different algorithms for our regression model: **Support Vector Regression (SVR)**, **K-Nearest Neighbors**, **Decision Trees** and **Neural Networks (Multi-layer Perceptron)**.

At last, we tried to reduce the dimension of our dataset by using Principal Component Analysis (PCA) which combines highly correlated variables together to form a smaller number of an artificial set of variables that account for the most variance in the data.

## Experimental Evaluation
---
In this section, we will try to provide a step-by-step explanation of our approach and show the way different factors shaped our implementation along the way. 

Starting with the necessary imports:


```python
import math
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_predict
from sklearn import preprocessing, linear_model, neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from matplotlib import style
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
```


```python
# Reads CVS file to Data Frame
data = pd.read_csv('Global_Mobility_Report.csv')
```

After reading the input file, we proceed to select and analyze its data. Since we will predict the number of cases of CoV-19 and these might depend on country-specific factors, it may be relevant to focus on a specific region in order to effectively and impartially compare the performance of the algorithms. Being one of the most affected countries in the world, the US seemed like a good candidate for this research.


```python
country = 'US'
data = data.loc[(data.country == country)]
data = data.reset_index(drop=True)
# If we were taking the country column into account,
# it would intersting to take care of the Dummy Variable Trap.
# Alternatively, we chose to eliminate these columns completely.
data = data.drop(columns = ['country','iso'])
data
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
      <th>date</th>
      <th>grocery_pharmacy</th>
      <th>parks</th>
      <th>residential</th>
      <th>retail_recreation</th>
      <th>transit_stations</th>
      <th>workplaces</th>
      <th>total_cases</th>
      <th>fatalities</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/23/2020</td>
      <td>3.124</td>
      <td>22.982</td>
      <td>-0.823</td>
      <td>6.813</td>
      <td>4.992</td>
      <td>2.238</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/24/2020</td>
      <td>0.717</td>
      <td>9.233</td>
      <td>0.040</td>
      <td>1.709</td>
      <td>1.285</td>
      <td>2.776</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/25/2020</td>
      <td>1.222</td>
      <td>9.106</td>
      <td>-0.138</td>
      <td>4.031</td>
      <td>2.155</td>
      <td>1.899</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/26/2020</td>
      <td>2.464</td>
      <td>5.209</td>
      <td>-0.630</td>
      <td>7.340</td>
      <td>3.498</td>
      <td>2.198</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/27/2020</td>
      <td>3.429</td>
      <td>12.251</td>
      <td>-0.459</td>
      <td>7.503</td>
      <td>4.013</td>
      <td>1.834</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2/28/2020</td>
      <td>3.392</td>
      <td>9.685</td>
      <td>-1.284</td>
      <td>7.996</td>
      <td>5.340</td>
      <td>2.428</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2/29/2020</td>
      <td>7.349</td>
      <td>20.717</td>
      <td>-1.858</td>
      <td>11.518</td>
      <td>7.367</td>
      <td>4.403</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3/1/2020</td>
      <td>8.816</td>
      <td>17.755</td>
      <td>-1.490</td>
      <td>12.864</td>
      <td>6.691</td>
      <td>2.954</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3/2/2020</td>
      <td>6.123</td>
      <td>10.253</td>
      <td>-0.571</td>
      <td>7.283</td>
      <td>1.694</td>
      <td>2.979</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3/3/2020</td>
      <td>9.870</td>
      <td>20.048</td>
      <td>-0.914</td>
      <td>10.780</td>
      <td>2.976</td>
      <td>2.024</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3/4/2020</td>
      <td>6.172</td>
      <td>17.146</td>
      <td>-0.653</td>
      <td>8.295</td>
      <td>2.316</td>
      <td>2.578</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3/5/2020</td>
      <td>7.314</td>
      <td>21.099</td>
      <td>-0.665</td>
      <td>7.989</td>
      <td>2.413</td>
      <td>2.518</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3/6/2020</td>
      <td>2.710</td>
      <td>11.630</td>
      <td>-0.558</td>
      <td>5.342</td>
      <td>1.701</td>
      <td>2.007</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3/7/2020</td>
      <td>8.366</td>
      <td>29.301</td>
      <td>-1.319</td>
      <td>9.604</td>
      <td>6.718</td>
      <td>4.586</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3/8/2020</td>
      <td>7.457</td>
      <td>40.308</td>
      <td>-1.065</td>
      <td>9.783</td>
      <td>5.383</td>
      <td>1.680</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3/9/2020</td>
      <td>6.474</td>
      <td>31.053</td>
      <td>0.242</td>
      <td>5.875</td>
      <td>-0.971</td>
      <td>-0.471</td>
      <td>0</td>
      <td>0</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3/10/2020</td>
      <td>7.467</td>
      <td>18.240</td>
      <td>0.625</td>
      <td>6.129</td>
      <td>-2.587</td>
      <td>-1.549</td>
      <td>892</td>
      <td>28</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3/11/2020</td>
      <td>10.072</td>
      <td>26.313</td>
      <td>0.185</td>
      <td>7.546</td>
      <td>-2.158</td>
      <td>-1.159</td>
      <td>1214</td>
      <td>36</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3/12/2020</td>
      <td>22.608</td>
      <td>19.057</td>
      <td>1.093</td>
      <td>6.366</td>
      <td>-6.292</td>
      <td>-2.903</td>
      <td>1596</td>
      <td>40</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3/13/2020</td>
      <td>25.884</td>
      <td>4.332</td>
      <td>2.836</td>
      <td>1.514</td>
      <td>-10.247</td>
      <td>-6.855</td>
      <td>2112</td>
      <td>47</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3/14/2020</td>
      <td>18.339</td>
      <td>2.665</td>
      <td>3.105</td>
      <td>-6.332</td>
      <td>-9.845</td>
      <td>-0.989</td>
      <td>2658</td>
      <td>54</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3/15/2020</td>
      <td>11.804</td>
      <td>4.734</td>
      <td>3.391</td>
      <td>-9.351</td>
      <td>-13.214</td>
      <td>-6.666</td>
      <td>3431</td>
      <td>63</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3/16/2020</td>
      <td>21.230</td>
      <td>-1.000</td>
      <td>7.600</td>
      <td>-7.559</td>
      <td>-22.047</td>
      <td>-19.867</td>
      <td>4565</td>
      <td>85</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3/17/2020</td>
      <td>13.762</td>
      <td>7.847</td>
      <td>10.739</td>
      <td>-18.104</td>
      <td>-26.256</td>
      <td>-26.753</td>
      <td>6353</td>
      <td>108</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3/18/2020</td>
      <td>7.594</td>
      <td>6.272</td>
      <td>12.635</td>
      <td>-23.678</td>
      <td>-30.283</td>
      <td>-30.611</td>
      <td>7715</td>
      <td>118</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3/19/2020</td>
      <td>6.668</td>
      <td>1.505</td>
      <td>14.618</td>
      <td>-27.935</td>
      <td>-35.502</td>
      <td>-33.412</td>
      <td>13608</td>
      <td>200</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3/20/2020</td>
      <td>5.161</td>
      <td>-2.340</td>
      <td>16.226</td>
      <td>-31.570</td>
      <td>-36.831</td>
      <td>-35.535</td>
      <td>19025</td>
      <td>244</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3/21/2020</td>
      <td>-1.731</td>
      <td>-2.889</td>
      <td>11.522</td>
      <td>-39.651</td>
      <td>-35.842</td>
      <td>-23.993</td>
      <td>25435</td>
      <td>307</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3/22/2020</td>
      <td>-13.413</td>
      <td>-11.594</td>
      <td>10.516</td>
      <td>-43.665</td>
      <td>-43.575</td>
      <td>-31.217</td>
      <td>33765</td>
      <td>426</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3/23/2020</td>
      <td>-10.195</td>
      <td>-22.538</td>
      <td>17.284</td>
      <td>-37.508</td>
      <td>-47.465</td>
      <td>-41.188</td>
      <td>43586</td>
      <td>551</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3/24/2020</td>
      <td>-11.921</td>
      <td>-9.906</td>
      <td>18.785</td>
      <td>-39.158</td>
      <td>-45.851</td>
      <td>-44.553</td>
      <td>53659</td>
      <td>705</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3/25/2020</td>
      <td>-14.705</td>
      <td>-7.404</td>
      <td>19.285</td>
      <td>-40.020</td>
      <td>-47.242</td>
      <td>-45.844</td>
      <td>65701</td>
      <td>941</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>32</th>
      <td>3/26/2020</td>
      <td>-13.802</td>
      <td>2.884</td>
      <td>19.859</td>
      <td>-40.277</td>
      <td>-47.915</td>
      <td>-46.601</td>
      <td>83759</td>
      <td>1208</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>33</th>
      <td>3/27/2020</td>
      <td>-16.054</td>
      <td>-7.085</td>
      <td>21.004</td>
      <td>-42.101</td>
      <td>-49.006</td>
      <td>-46.307</td>
      <td>101580</td>
      <td>1578</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>34</th>
      <td>3/28/2020</td>
      <td>-18.931</td>
      <td>-25.226</td>
      <td>14.636</td>
      <td>-47.734</td>
      <td>-49.306</td>
      <td>-33.727</td>
      <td>121313</td>
      <td>2023</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3/29/2020</td>
      <td>-22.299</td>
      <td>-18.639</td>
      <td>12.185</td>
      <td>-46.961</td>
      <td>-51.399</td>
      <td>-37.647</td>
      <td>140757</td>
      <td>2464</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3/30/2020</td>
      <td>-17.564</td>
      <td>-14.570</td>
      <td>18.984</td>
      <td>-40.197</td>
      <td>-50.356</td>
      <td>-47.603</td>
      <td>161679</td>
      <td>2975</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3/31/2020</td>
      <td>-15.693</td>
      <td>-16.072</td>
      <td>20.218</td>
      <td>-39.983</td>
      <td>-49.420</td>
      <td>-49.328</td>
      <td>188018</td>
      <td>3870</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4/1/2020</td>
      <td>-10.815</td>
      <td>-7.344</td>
      <td>19.779</td>
      <td>-37.299</td>
      <td>-47.939</td>
      <td>-49.233</td>
      <td>213214</td>
      <td>4753</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4/2/2020</td>
      <td>-11.209</td>
      <td>-8.520</td>
      <td>20.952</td>
      <td>-39.891</td>
      <td>-50.559</td>
      <td>-49.853</td>
      <td>243441</td>
      <td>5922</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4/3/2020</td>
      <td>-11.795</td>
      <td>-20.610</td>
      <td>22.228</td>
      <td>-41.889</td>
      <td>-51.329</td>
      <td>-49.575</td>
      <td>275426</td>
      <td>7083</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4/4/2020</td>
      <td>-14.975</td>
      <td>-23.779</td>
      <td>15.164</td>
      <td>-48.466</td>
      <td>-50.208</td>
      <td>-36.657</td>
      <td>308693</td>
      <td>8403</td>
      <td>329064917</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4/5/2020</td>
      <td>-20.249</td>
      <td>-20.271</td>
      <td>12.550</td>
      <td>-48.911</td>
      <td>-53.680</td>
      <td>-40.031</td>
      <td>336912</td>
      <td>9615</td>
      <td>329064917</td>
    </tr>
  </tbody>
</table>
</div>



Extraction of new relevant columns:


```python
# Calculates daily percentual changes in number of fatalities
data['fatalities_percent_change'] = pd.Series([])

for i in range(0,len(data['fatalities'])):
    if i != 0 and data['fatalities'][i-1] != 0:
        data['fatalities_percent_change'][i]= (data['fatalities'][i] - data['fatalities'][i-1]) * 100 /data['fatalities'][i-1]
    else:data['fatalities_percent_change'][i] = 0

# Calculates daily percentual changes in number of cases
data['cases_percent_change'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    if i != 0 and data['total_cases'][i-1] != 0:
        data['cases_percent_change'][i] = (data['total_cases'][i] - data['total_cases'][i-1]) * 100 /data['total_cases'][i-1]
    else:data['cases_percent_change'][i] = 0

# Calculates cases of covid per million inhabitants
data['cases_per_million'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    millions = data['population'][i]/1000000
    data['cases_per_million'][i]= data['total_cases'][i]/millions

# Drop unnecessary columns from data set
data = data.drop(columns = ['fatalities'])
data = data.drop(columns = ['total_cases'])
data = data.drop(columns = ['population'])

data
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
      <th>date</th>
      <th>grocery_pharmacy</th>
      <th>parks</th>
      <th>residential</th>
      <th>retail_recreation</th>
      <th>transit_stations</th>
      <th>workplaces</th>
      <th>fatalities_percent_change</th>
      <th>cases_percent_change</th>
      <th>cases_per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/23/2020</td>
      <td>3.124</td>
      <td>22.982</td>
      <td>-0.823</td>
      <td>6.813</td>
      <td>4.992</td>
      <td>2.238</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/24/2020</td>
      <td>0.717</td>
      <td>9.233</td>
      <td>0.040</td>
      <td>1.709</td>
      <td>1.285</td>
      <td>2.776</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/25/2020</td>
      <td>1.222</td>
      <td>9.106</td>
      <td>-0.138</td>
      <td>4.031</td>
      <td>2.155</td>
      <td>1.899</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/26/2020</td>
      <td>2.464</td>
      <td>5.209</td>
      <td>-0.630</td>
      <td>7.340</td>
      <td>3.498</td>
      <td>2.198</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/27/2020</td>
      <td>3.429</td>
      <td>12.251</td>
      <td>-0.459</td>
      <td>7.503</td>
      <td>4.013</td>
      <td>1.834</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2/28/2020</td>
      <td>3.392</td>
      <td>9.685</td>
      <td>-1.284</td>
      <td>7.996</td>
      <td>5.340</td>
      <td>2.428</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2/29/2020</td>
      <td>7.349</td>
      <td>20.717</td>
      <td>-1.858</td>
      <td>11.518</td>
      <td>7.367</td>
      <td>4.403</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3/1/2020</td>
      <td>8.816</td>
      <td>17.755</td>
      <td>-1.490</td>
      <td>12.864</td>
      <td>6.691</td>
      <td>2.954</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3/2/2020</td>
      <td>6.123</td>
      <td>10.253</td>
      <td>-0.571</td>
      <td>7.283</td>
      <td>1.694</td>
      <td>2.979</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3/3/2020</td>
      <td>9.870</td>
      <td>20.048</td>
      <td>-0.914</td>
      <td>10.780</td>
      <td>2.976</td>
      <td>2.024</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3/4/2020</td>
      <td>6.172</td>
      <td>17.146</td>
      <td>-0.653</td>
      <td>8.295</td>
      <td>2.316</td>
      <td>2.578</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3/5/2020</td>
      <td>7.314</td>
      <td>21.099</td>
      <td>-0.665</td>
      <td>7.989</td>
      <td>2.413</td>
      <td>2.518</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3/6/2020</td>
      <td>2.710</td>
      <td>11.630</td>
      <td>-0.558</td>
      <td>5.342</td>
      <td>1.701</td>
      <td>2.007</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3/7/2020</td>
      <td>8.366</td>
      <td>29.301</td>
      <td>-1.319</td>
      <td>9.604</td>
      <td>6.718</td>
      <td>4.586</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3/8/2020</td>
      <td>7.457</td>
      <td>40.308</td>
      <td>-1.065</td>
      <td>9.783</td>
      <td>5.383</td>
      <td>1.680</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3/9/2020</td>
      <td>6.474</td>
      <td>31.053</td>
      <td>0.242</td>
      <td>5.875</td>
      <td>-0.971</td>
      <td>-0.471</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3/10/2020</td>
      <td>7.467</td>
      <td>18.240</td>
      <td>0.625</td>
      <td>6.129</td>
      <td>-2.587</td>
      <td>-1.549</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.710711</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3/11/2020</td>
      <td>10.072</td>
      <td>26.313</td>
      <td>0.185</td>
      <td>7.546</td>
      <td>-2.158</td>
      <td>-1.159</td>
      <td>28.571429</td>
      <td>36.098655</td>
      <td>3.689242</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3/12/2020</td>
      <td>22.608</td>
      <td>19.057</td>
      <td>1.093</td>
      <td>6.366</td>
      <td>-6.292</td>
      <td>-2.903</td>
      <td>11.111111</td>
      <td>31.466227</td>
      <td>4.850107</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3/13/2020</td>
      <td>25.884</td>
      <td>4.332</td>
      <td>2.836</td>
      <td>1.514</td>
      <td>-10.247</td>
      <td>-6.855</td>
      <td>17.500000</td>
      <td>32.330827</td>
      <td>6.418186</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3/14/2020</td>
      <td>18.339</td>
      <td>2.665</td>
      <td>3.105</td>
      <td>-6.332</td>
      <td>-9.845</td>
      <td>-0.989</td>
      <td>14.893617</td>
      <td>25.852273</td>
      <td>8.077434</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3/15/2020</td>
      <td>11.804</td>
      <td>4.734</td>
      <td>3.391</td>
      <td>-9.351</td>
      <td>-13.214</td>
      <td>-6.666</td>
      <td>16.666667</td>
      <td>29.082017</td>
      <td>10.426514</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3/16/2020</td>
      <td>21.230</td>
      <td>-1.000</td>
      <td>7.600</td>
      <td>-7.559</td>
      <td>-22.047</td>
      <td>-19.867</td>
      <td>34.920635</td>
      <td>33.051588</td>
      <td>13.872643</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3/17/2020</td>
      <td>13.762</td>
      <td>7.847</td>
      <td>10.739</td>
      <td>-18.104</td>
      <td>-26.256</td>
      <td>-26.753</td>
      <td>27.058824</td>
      <td>39.167579</td>
      <td>19.306221</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3/18/2020</td>
      <td>7.594</td>
      <td>6.272</td>
      <td>12.635</td>
      <td>-23.678</td>
      <td>-30.283</td>
      <td>-30.611</td>
      <td>9.259259</td>
      <td>21.438690</td>
      <td>23.445222</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3/19/2020</td>
      <td>6.668</td>
      <td>1.505</td>
      <td>14.618</td>
      <td>-27.935</td>
      <td>-35.502</td>
      <td>-33.412</td>
      <td>69.491525</td>
      <td>76.383668</td>
      <td>41.353542</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3/20/2020</td>
      <td>5.161</td>
      <td>-2.340</td>
      <td>16.226</td>
      <td>-31.570</td>
      <td>-36.831</td>
      <td>-35.535</td>
      <td>22.000000</td>
      <td>39.807466</td>
      <td>57.815340</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3/21/2020</td>
      <td>-1.731</td>
      <td>-2.889</td>
      <td>11.522</td>
      <td>-39.651</td>
      <td>-35.842</td>
      <td>-23.993</td>
      <td>25.819672</td>
      <td>33.692510</td>
      <td>77.294779</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3/22/2020</td>
      <td>-13.413</td>
      <td>-11.594</td>
      <td>10.516</td>
      <td>-43.665</td>
      <td>-43.575</td>
      <td>-31.217</td>
      <td>38.762215</td>
      <td>32.750147</td>
      <td>102.608933</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3/23/2020</td>
      <td>-10.195</td>
      <td>-22.538</td>
      <td>17.284</td>
      <td>-37.508</td>
      <td>-47.465</td>
      <td>-41.188</td>
      <td>29.342723</td>
      <td>29.086332</td>
      <td>132.454108</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3/24/2020</td>
      <td>-11.921</td>
      <td>-9.906</td>
      <td>18.785</td>
      <td>-39.158</td>
      <td>-45.851</td>
      <td>-44.553</td>
      <td>27.949183</td>
      <td>23.110632</td>
      <td>163.065089</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3/25/2020</td>
      <td>-14.705</td>
      <td>-7.404</td>
      <td>19.285</td>
      <td>-40.020</td>
      <td>-47.242</td>
      <td>-45.844</td>
      <td>33.475177</td>
      <td>22.441715</td>
      <td>199.659692</td>
    </tr>
    <tr>
      <th>32</th>
      <td>3/26/2020</td>
      <td>-13.802</td>
      <td>2.884</td>
      <td>19.859</td>
      <td>-40.277</td>
      <td>-47.915</td>
      <td>-46.601</td>
      <td>28.374070</td>
      <td>27.485122</td>
      <td>254.536402</td>
    </tr>
    <tr>
      <th>33</th>
      <td>3/27/2020</td>
      <td>-16.054</td>
      <td>-7.085</td>
      <td>21.004</td>
      <td>-42.101</td>
      <td>-49.006</td>
      <td>-46.307</td>
      <td>30.629139</td>
      <td>21.276520</td>
      <td>308.692889</td>
    </tr>
    <tr>
      <th>34</th>
      <td>3/28/2020</td>
      <td>-18.931</td>
      <td>-25.226</td>
      <td>14.636</td>
      <td>-47.734</td>
      <td>-49.306</td>
      <td>-33.727</td>
      <td>28.200253</td>
      <td>19.426068</td>
      <td>368.659780</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3/29/2020</td>
      <td>-22.299</td>
      <td>-18.639</td>
      <td>12.185</td>
      <td>-46.961</td>
      <td>-51.399</td>
      <td>-37.647</td>
      <td>21.799308</td>
      <td>16.027961</td>
      <td>427.748425</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3/30/2020</td>
      <td>-17.564</td>
      <td>-14.570</td>
      <td>18.984</td>
      <td>-40.197</td>
      <td>-50.356</td>
      <td>-47.603</td>
      <td>20.738636</td>
      <td>14.863914</td>
      <td>491.328585</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3/31/2020</td>
      <td>-15.693</td>
      <td>-16.072</td>
      <td>20.218</td>
      <td>-39.983</td>
      <td>-49.420</td>
      <td>-49.328</td>
      <td>30.084034</td>
      <td>16.290922</td>
      <td>571.370542</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4/1/2020</td>
      <td>-10.815</td>
      <td>-7.344</td>
      <td>19.779</td>
      <td>-37.299</td>
      <td>-47.939</td>
      <td>-49.233</td>
      <td>22.816537</td>
      <td>13.400845</td>
      <td>647.939020</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4/2/2020</td>
      <td>-11.209</td>
      <td>-8.520</td>
      <td>20.952</td>
      <td>-39.891</td>
      <td>-50.559</td>
      <td>-49.853</td>
      <td>24.594993</td>
      <td>14.176836</td>
      <td>739.796276</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4/3/2020</td>
      <td>-11.795</td>
      <td>-20.610</td>
      <td>22.228</td>
      <td>-41.889</td>
      <td>-51.329</td>
      <td>-49.575</td>
      <td>19.604863</td>
      <td>13.138707</td>
      <td>836.995941</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4/4/2020</td>
      <td>-14.975</td>
      <td>-23.779</td>
      <td>15.164</td>
      <td>-48.466</td>
      <td>-50.208</td>
      <td>-36.657</td>
      <td>18.636171</td>
      <td>12.078380</td>
      <td>938.091495</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4/5/2020</td>
      <td>-20.249</td>
      <td>-20.271</td>
      <td>12.550</td>
      <td>-48.911</td>
      <td>-53.680</td>
      <td>-40.031</td>
      <td>14.423420</td>
      <td>9.141445</td>
      <td>1023.846611</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We start by looking for the day of the first confirmed cases.
starting_day = pd.to_datetime(data['date'][0])
day_index = 0
for day_index in range(0,len(data['cases_per_million'])):
    if data['cases_per_million'][day_index] != 0:
        starting_day = pd.to_datetime(data['date'][day_index -1])
        break

dates = data['date']
date_format = [pd.to_datetime(d) for d in dates]

# A first look at the evolution of the number of confirmed cases per million 
# and how the mobility (namely retail and recreation) was affected.
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
cases_per_million_plot, = ax.plot(date_format, data['cases_per_million'])
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title=("Evolution of CoV-19 Cases per Million (%s)" % (country)))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.axvline(starting_day, c="yellow", zorder=0)
ax.tick_params(axis='y', labelcolor= "blue")
ax2 = ax.twinx()  
ax2.set_ylabel('Retail Mobility Change')
grocery_pharmacy, = ax2.plot(date_format, data['grocery_pharmacy'], color = "darkred")
parks, = ax2.plot(date_format, data['parks'], color = "crimson")
residential, = ax2.plot(date_format, data['residential'], color = "orangered")
retail_recreation, = ax2.plot(date_format, data['retail_recreation'], color = "tomato")
transit_stations, = ax2.plot(date_format, data['transit_stations'], color = "coral")

ax.legend((grocery_pharmacy, parks, residential, retail_recreation, transit_stations, cases_per_million_plot), ('Grocery & Pharmacy', 'Parks', 'Residential', 'Retail & Recreation', 'Transit Stations','Number of CoV-19 Cases per Million'), loc='upper left', shadow=False)

ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax2.tick_params(axis='y', labelcolor= "red")
plt.show()
```


![svg](res/output_7_0.svg)


As we can see from the graph above, when the first cases of CoV-19 were registered, the overall mobility dropped considerably, with exception of the residential trendline which, given the context of our dataset, increases as expected. For this reason, we will only consider data starting from this inflection point on.

That being said, we can now define the Total Number of Cases per Million as our **target variable** and isolate it from our **independent variables**. 


```python
X = data.iloc[day_index:,1:9]
Y = data.iloc[day_index:,9:]
X
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
      <th>grocery_pharmacy</th>
      <th>parks</th>
      <th>residential</th>
      <th>retail_recreation</th>
      <th>transit_stations</th>
      <th>workplaces</th>
      <th>fatalities_percent_change</th>
      <th>cases_percent_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>7.467</td>
      <td>18.240</td>
      <td>0.625</td>
      <td>6.129</td>
      <td>-2.587</td>
      <td>-1.549</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10.072</td>
      <td>26.313</td>
      <td>0.185</td>
      <td>7.546</td>
      <td>-2.158</td>
      <td>-1.159</td>
      <td>28.571429</td>
      <td>36.098655</td>
    </tr>
    <tr>
      <th>18</th>
      <td>22.608</td>
      <td>19.057</td>
      <td>1.093</td>
      <td>6.366</td>
      <td>-6.292</td>
      <td>-2.903</td>
      <td>11.111111</td>
      <td>31.466227</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25.884</td>
      <td>4.332</td>
      <td>2.836</td>
      <td>1.514</td>
      <td>-10.247</td>
      <td>-6.855</td>
      <td>17.500000</td>
      <td>32.330827</td>
    </tr>
    <tr>
      <th>20</th>
      <td>18.339</td>
      <td>2.665</td>
      <td>3.105</td>
      <td>-6.332</td>
      <td>-9.845</td>
      <td>-0.989</td>
      <td>14.893617</td>
      <td>25.852273</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11.804</td>
      <td>4.734</td>
      <td>3.391</td>
      <td>-9.351</td>
      <td>-13.214</td>
      <td>-6.666</td>
      <td>16.666667</td>
      <td>29.082017</td>
    </tr>
    <tr>
      <th>22</th>
      <td>21.230</td>
      <td>-1.000</td>
      <td>7.600</td>
      <td>-7.559</td>
      <td>-22.047</td>
      <td>-19.867</td>
      <td>34.920635</td>
      <td>33.051588</td>
    </tr>
    <tr>
      <th>23</th>
      <td>13.762</td>
      <td>7.847</td>
      <td>10.739</td>
      <td>-18.104</td>
      <td>-26.256</td>
      <td>-26.753</td>
      <td>27.058824</td>
      <td>39.167579</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7.594</td>
      <td>6.272</td>
      <td>12.635</td>
      <td>-23.678</td>
      <td>-30.283</td>
      <td>-30.611</td>
      <td>9.259259</td>
      <td>21.438690</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6.668</td>
      <td>1.505</td>
      <td>14.618</td>
      <td>-27.935</td>
      <td>-35.502</td>
      <td>-33.412</td>
      <td>69.491525</td>
      <td>76.383668</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5.161</td>
      <td>-2.340</td>
      <td>16.226</td>
      <td>-31.570</td>
      <td>-36.831</td>
      <td>-35.535</td>
      <td>22.000000</td>
      <td>39.807466</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-1.731</td>
      <td>-2.889</td>
      <td>11.522</td>
      <td>-39.651</td>
      <td>-35.842</td>
      <td>-23.993</td>
      <td>25.819672</td>
      <td>33.692510</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-13.413</td>
      <td>-11.594</td>
      <td>10.516</td>
      <td>-43.665</td>
      <td>-43.575</td>
      <td>-31.217</td>
      <td>38.762215</td>
      <td>32.750147</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-10.195</td>
      <td>-22.538</td>
      <td>17.284</td>
      <td>-37.508</td>
      <td>-47.465</td>
      <td>-41.188</td>
      <td>29.342723</td>
      <td>29.086332</td>
    </tr>
    <tr>
      <th>30</th>
      <td>-11.921</td>
      <td>-9.906</td>
      <td>18.785</td>
      <td>-39.158</td>
      <td>-45.851</td>
      <td>-44.553</td>
      <td>27.949183</td>
      <td>23.110632</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-14.705</td>
      <td>-7.404</td>
      <td>19.285</td>
      <td>-40.020</td>
      <td>-47.242</td>
      <td>-45.844</td>
      <td>33.475177</td>
      <td>22.441715</td>
    </tr>
    <tr>
      <th>32</th>
      <td>-13.802</td>
      <td>2.884</td>
      <td>19.859</td>
      <td>-40.277</td>
      <td>-47.915</td>
      <td>-46.601</td>
      <td>28.374070</td>
      <td>27.485122</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-16.054</td>
      <td>-7.085</td>
      <td>21.004</td>
      <td>-42.101</td>
      <td>-49.006</td>
      <td>-46.307</td>
      <td>30.629139</td>
      <td>21.276520</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-18.931</td>
      <td>-25.226</td>
      <td>14.636</td>
      <td>-47.734</td>
      <td>-49.306</td>
      <td>-33.727</td>
      <td>28.200253</td>
      <td>19.426068</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-22.299</td>
      <td>-18.639</td>
      <td>12.185</td>
      <td>-46.961</td>
      <td>-51.399</td>
      <td>-37.647</td>
      <td>21.799308</td>
      <td>16.027961</td>
    </tr>
    <tr>
      <th>36</th>
      <td>-17.564</td>
      <td>-14.570</td>
      <td>18.984</td>
      <td>-40.197</td>
      <td>-50.356</td>
      <td>-47.603</td>
      <td>20.738636</td>
      <td>14.863914</td>
    </tr>
    <tr>
      <th>37</th>
      <td>-15.693</td>
      <td>-16.072</td>
      <td>20.218</td>
      <td>-39.983</td>
      <td>-49.420</td>
      <td>-49.328</td>
      <td>30.084034</td>
      <td>16.290922</td>
    </tr>
    <tr>
      <th>38</th>
      <td>-10.815</td>
      <td>-7.344</td>
      <td>19.779</td>
      <td>-37.299</td>
      <td>-47.939</td>
      <td>-49.233</td>
      <td>22.816537</td>
      <td>13.400845</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-11.209</td>
      <td>-8.520</td>
      <td>20.952</td>
      <td>-39.891</td>
      <td>-50.559</td>
      <td>-49.853</td>
      <td>24.594993</td>
      <td>14.176836</td>
    </tr>
    <tr>
      <th>40</th>
      <td>-11.795</td>
      <td>-20.610</td>
      <td>22.228</td>
      <td>-41.889</td>
      <td>-51.329</td>
      <td>-49.575</td>
      <td>19.604863</td>
      <td>13.138707</td>
    </tr>
    <tr>
      <th>41</th>
      <td>-14.975</td>
      <td>-23.779</td>
      <td>15.164</td>
      <td>-48.466</td>
      <td>-50.208</td>
      <td>-36.657</td>
      <td>18.636171</td>
      <td>12.078380</td>
    </tr>
    <tr>
      <th>42</th>
      <td>-20.249</td>
      <td>-20.271</td>
      <td>12.550</td>
      <td>-48.911</td>
      <td>-53.680</td>
      <td>-40.031</td>
      <td>14.423420</td>
      <td>9.141445</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replaces NaN values present in the new columns by 0 (Zero Inputation)
X['fatalities_percent_change'] = X['fatalities_percent_change'].replace(np.NaN, 0)
X['cases_percent_change'] = X['cases_percent_change'].replace(np.NaN, 0)
Y['cases_per_million'] = Y['cases_per_million'].replace(np.NaN, 0)

# Fills the empty values taking mean values present in each column (Mean Inputation)
X = X.replace("", np.NaN)
Y = Y.replace("", np.NaN)
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

Y
```




    array([[   2.71071133],
           [   3.68924166],
           [   4.85010683],
           [   6.41818648],
           [   8.07743355],
           [  10.42651411],
           [  13.87264264],
           [  19.30622097],
           [  23.4452219 ],
           [  41.35354241],
           [  57.81533982],
           [  77.29477889],
           [ 102.60893294],
           [ 132.45410783],
           [ 163.06508907],
           [ 199.65969207],
           [ 254.53640201],
           [ 308.69288931],
           [ 368.65978028],
           [ 427.74842509],
           [ 491.32858487],
           [ 571.37054206],
           [ 647.93902049],
           [ 739.79627552],
           [ 836.99594144],
           [ 938.09149518],
           [1023.84661079]])



To prevent a feature that has a variance that is orders of magnitude larger than others, from dominating the objective function and make the estimator unable to learn from other features correctly as expected, it is wise to perform standardization of such features.


```python
# Standardization using Gaussian Normal Distribution
X = preprocessing.scale(X)
```

Once pre-processed, our data can now be divided into subsets meant for training and testing our models. Usually, it would be very convenient to make use of K-Fold Cross-Validation (recommended for big datasets), since it provides a better way of partitioning the data into the mentioned subsets, leading to greater results. Unfortunately, given the small dimension of our dataset, we were not able to obtain satisfactory enough results in order to use this method. Thus, we decided to apply Train/Test Split (80%/20%) which randomly selects the input and output data as training or testing data.


```python
# Create training and testing vars
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
```

We are now ready to start training and testing our models.

### Support Vector Regression

We start by using the SVR algorithm, which is an extension of the Support Vector Machine algorithm, usually used in Classification models, but applied to Regression. The **scikit-learn** library provides several alternatives for kernel functions, namely **Radial Basis** (the default option according to the documentation), **Linear**, or **Polynomial**. Besides being relatively memory efficient, the SVR algorithm works well with high dimensional spaces and small datasets without too much noise. Making it a good contender for this problem.
In this section, we will be comparing the performance of these kernels to the **Ordinary Least Squares Linear Regression** (OLS). 



```python
# Linear Regression Model
lm = linear_model.LinearRegression()

# Suport Vector Regression Models
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

# Fitting/Training models
model = lm.fit(X_train, Y_train)
svr_lin_model = svr_lin.fit(X_train, Y_train)
svr_rbf_model = svr_rbf.fit(X_train, Y_train)
svr_poly_model = svr_poly.fit(X_train, Y_train)

# Obtaining predictions from models
linear_predictions = lm.predict(X)
svr_lin_predictions = svr_lin_model.predict(X)
svr_rbf_predictions = svr_rbf_model.predict(X)
svr_poly_predictions = svr_poly_model.predict(X)

# Testing models
linear_score = model.score(X_test, Y_test)
svr_lin_score = svr_lin_model.score(X_test, Y_test)
svr_rbf_score = svr_rbf_model.score(X_test, Y_test)
svr_poly_score = svr_poly_model.score(X_test, Y_test)

print("Linear Regression Score:", linear_score)
print("Linear SVR Score: %s" % (svr_lin_score))
print("RBF SVR Score: %s" % (svr_rbf_score))
print("Polynomial SVR Score: %s" % (svr_poly_score))
```

    Linear Regression Score: 0.3756860295490796
    Linear SVR Score: 0.56572413789943
    RBF SVR Score: 0.6789530397892976
    Polynomial SVR Score: 0.8841793709604677
    


```python
# Visual Comparison of the predictions obtained by different 
# parameterizations of the SVR model compared to a Linear Regression model
dates = data['date']
date_format = [pd.to_datetime(d) for d in dates]
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
ax.scatter(date_format[day_index:], data['cases_per_million'][day_index:])
linear_plot, = ax.plot(date_format[day_index:], linear_predictions)
svr_poly_plot, = ax.plot(date_format[day_index:], svr_poly_predictions)
svr_lin_plot, = ax.plot(date_format[day_index:], svr_lin_predictions)
svr_rbf_plot, = ax.plot(date_format[day_index:], svr_rbf_predictions)
ax.legend((linear_plot, svr_poly_plot, svr_lin_plot, svr_rbf_plot), ('Linear Regression', 'SVR Polynomial Kernel','SVR Linear Kernel', 'SVR Radial Basis Function Kernel'), loc='upper left', shadow=False)

ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title=("Evolution of CoV-19 Cases per Million (%s)" % (country)))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.tick_params(axis='y', labelcolor= "blue")
```


![svg](res/output_17_0.svg)


As we can see by the scores obtained by the comparison of the predictions with the labeled data in several experiments, the performance of the SVR linear kernel tends to be worse or as goods as OLS, whereas the remaining kernels tend to be better for this particular dataset.

### K-Nearest Neighbor

We proceed to apply the K-Nearest Neighbor algorithm. Once again, we are given two alternatives for the weights specified for the "neighbors". Using a uniform weight distribution, each of the k-neighbor's weight is taken into account equally. On the other hand, when considering the distance between these neighbors, the weight decreases with the distance.


```python
n_neighbors = 3
# K-Nearest Neighbor Models
knn_uniform = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn_distance = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')

# Fitting/Training and obtaining predictions from models
knn_predictions_uniform = knn_uniform.fit(X_train, Y_train).predict(X)
knn_predictions_distance = knn_distance.fit(X_train, Y_train).predict(X)

# Testing models
knn_uniform_score = knn_uniform.score(X_test, Y_test)
knn_distance_score = knn_distance.score(X_test, Y_test)

print("KNN Regression Model - Uniform:  %s"  % (knn_uniform_score))
print("KNN Regression Model - Distance:  %s"  % (knn_distance_score))

# Visual Comparison of the predictions obtained by different 
# parameterizations of the KNN model
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[day_index:], Y, color='darkorange', label='data')
unif, = ax.plot(date_format[day_index:], knn_predictions_uniform, color='navy')
dist, = ax.plot(date_format[day_index:], knn_predictions_distance, color='darkgreen')
ax.legend((unif, dist, labeled), ('Uniform', 'Distance','Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million (%s) : K-Nearest Neighbor (k = %i) " % (country, n_neighbors))
ax.axis('tight')
plt.tight_layout()
plt.show()
```

    KNN Regression Model - Uniform:  0.70197052312614
    KNN Regression Model - Distance:  0.6827051489302913
    


![svg](res/output_19_1.svg)


The results obtained with the parameterizations of this algorithm were very promising, with a slight advantage for the distance-based approach.

### Decision Tree Regression

Another worthy approach for this regression problem is the Decision Tree algorithm. It's important to observe the influence of the *max_depth* parameter of the model to see how it affects its performance. With that in mind, we chose to compare two DT models with *max_depth* values of 2 and 10.

Usually, it would be risky to use a high value for the *max_depth* parameter of a decision tree since it can lead to the model learning from noise present in the data and be prone to **overfit**. 


```python
# Decision Tree Models
dt_regressor_2 = DecisionTreeRegressor(max_depth=2)
dt_regressor_10 = DecisionTreeRegressor(max_depth=10)

# Fitting/Training models
dt_regressor_2.fit(X_train, Y_train)
dt_regressor_10.fit(X_train, Y_train)

# Obtaining predictions from models
max_depth2_prediction = dt_regressor_2.predict(X)
max_depth10_prediction = dt_regressor_10.predict(X)

# Testing models
dt_regressor_2_score = dt_regressor_2.score(X_test,Y_test)
dt_regressor_10_score = dt_regressor_10.score(X_test,Y_test)

print("DT2", dt_regressor_2_score)
print("DT10", dt_regressor_10_score)

# Plot the results
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[day_index:], Y, color='darkorange', label='data')
max_depth2, = ax.plot(date_format[day_index:], max_depth2_prediction, color='cornflowerblue')
max_depth10, = ax.plot(date_format[day_index:], max_depth10_prediction, color='yellowgreen')
ax.legend((max_depth2, max_depth10, labeled), ('max_depth-2', 'max_depth-10','Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million (%s) : Decision Tree Regression" % (country))
plt.tight_layout()
plt.show()
```

    DT2 0.9952366814531595
    DT10 0.9783387675294882
    


![svg](res/output_21_1.svg)


Fortunately, given that our data does not contain noise and was handled carefully beforehand, the use of a high *max_depth* parameter tends to be beneficial.

### Neural Network - Multi-layer Perceptron

Finally, we continue our study by using a Neural Network. For this, we are using a multi-layer perceptron which iteratively corrects its parameters according to the partial derivatives of the loss function.
The default solver *adam* works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, *lbfgs* can converge faster and perform better.


```python
hidden_layers = 5
# Fitting/Training Neural Network (MLP) Model
model = MLPRegressor(hidden_layers, validation_fraction = 0, solver='lbfgs').fit(X_train, Y_train)

# Obtaining predictions from models
NN_predict = model.predict(X)

# Testing models
NN_score = model.score(X_test, Y_test)
print('Score', NN_score)

# Visual Comparison of the predictions obtained by the NN model
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[day_index:], Y, color='darkorange', label='data')
NN_plot, = ax.plot(date_format[day_index:], NN_predict, color='cornflowerblue')
ax.legend((NN_plot,labeled), ('Perceptron\'s prediction', 'Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million (%s) : Neural network - Multi-layer Perceptron" % (country))
plt.tight_layout()
plt.show()
```

    Score 0.963035649571465
    


![svg](res/output_23_1.svg)


Although it requires a lot of computational power, especially for such a small dataset, it provides good and consistent results.

To better visualize the differences, in terms of scores, between the several algorithms we can look at the graph below.


```python
# Visual Comparison of the scores obtained by the different 
# models when predicting labeled data
_, ax = plt.subplots(figsize=(24, 10))
ax.bar('Linear Regression', linear_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('SVR Linear Regression', svr_lin_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('SVR Radial Basis Function Regression', svr_rbf_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('SVR Polynomial Regression', svr_poly_score, width=0.8, bottom=None, align='center', data=None)

ax.bar('K-Nearest-Neighbors (Uniform)', knn_uniform_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('K-Nearest-Neighbors (Distance)', knn_distance_score, width=0.8, bottom=None, align='center', data=None)

ax.bar('Decision Tree (Depth = 2)', dt_regressor_2_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('Decision Tree (Depth = 10)', dt_regressor_10_score, width=0.8, bottom=None, align='center', data=None)

ax.bar('Neural Network (MLP)', NN_score, width=0.8, bottom=None, align='center', data=None)

plt.xticks(rotation=90)
```








![svg](res/output_25_1.svg)


### Principal Component Analysis

PCA is a statistical procedure that uses an orthogonal transformation to convert a set of variables, possibly correlated, into a set of values ​​of variables not linearly correlated, the main components. The PCA must be applied to the training set to obtain a new coordinate system defined only by the proper vectors, which are statistically significant.



```python
pca = PCA(0.95)
principalComponents = pca.fit_transform(X) 
print(pca.explained_variance_ratio_)
principalDf = pd.DataFrame(data = principalComponents) 
print(pca.components_)
```

    [0.68247997 0.22296736 0.04696344]
    [[ 0.39289409  0.37283452 -0.39404191  0.41568115  0.42553647  0.40800669
      -0.11841312  0.13037019]
     [ 0.13027736  0.07523198  0.0954432  -0.02519027 -0.02923957 -0.06659996
       0.69299011  0.69433292]
     [-0.1337239  -0.66305254 -0.54063077 -0.19280261 -0.00926701  0.43545896
       0.1354905   0.0704039 ]]
    

Using this analysis, we obtain the principal components that make up for at least 95% of the variance ratio. These are the **Transit Stations**, **Daily Percentage Change of the Total Number of Cases**, and **Parks**, respectively.

### Another Approach
---
Knowing that different countries apply different policies and have different lifestyles, we believe it would be interesting to see the evolution of the average total number of cases per million amongst the 19 countries included in the dataset.


```python
# Reads CVS file to Data Frame
data = pd.read_csv('Global_Mobility_Report.csv')
data = data.drop(columns = ['country','iso'])

# Calculates daily percentual changes in number of fatalities
data['fatalities_percent_change'] = pd.Series([])
for i in range(0,len(data['fatalities'])):
    if i != 0 and data['fatalities'][i-1] != 0:
        data['fatalities_percent_change'][i]= (data['fatalities'][i] - data['fatalities'][i-1]) * 100 /data['fatalities'][i-1]
    else:data['fatalities_percent_change'][i] = 0

# Calculates daily percentual changes in number of cases
data['cases_percent_change'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    if i != 0 and data['total_cases'][i-1] != 0:
        data['cases_percent_change'][i]= (data['total_cases'][i] - data['total_cases'][i-1]) * 100 /data['total_cases'][i-1]
    else:data['cases_percent_change'][i] = 0

# Calculates cases of covid per million inhabitants
data['cases_per_million'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    millions = data['population'][i]/1000000
    data['cases_per_million'][i]= data['total_cases'][i]/millions

# Drop unnecessary columns from data set
data = data.drop(columns = ['fatalities'])
data = data.drop(columns = ['total_cases'])
data = data.drop(columns = ['population'])

dates = data['date']
date_format = [pd.to_datetime(d) for d in dates]

_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
cases_scatter1 = ax.scatter(date_format, data['cases_per_million'])
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title=("Evolution of CoV-19 Cases per Million (19 different countries)"))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.tick_params(axis='y', labelcolor= "blue")
ax2 = ax.twinx()  
ax2.set_ylabel('Retail and Recreation Mobility')
retail_scatter1 = ax2.scatter(date_format, data['retail_recreation'], color = "orange")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax2.tick_params(axis='y', labelcolor= "orange")
ax.legend((cases_scatter1,retail_scatter1), ('Retail and Recreation Mobility', 'Number of CoV-19 Cases per Million'), loc='upper left', shadow=False)
plt.show()
```


![svg](res/output_29_0.svg)


As we can see, both the number of confirmed cases and the mobility change vary considerably according to the country they represent. To predict how the average total number of cases per million would evolve, we are using a Neural Network similar to the previous one.


```python
_, ax = plt.subplots(figsize=(24, 10))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
cases_scatter2 = ax.scatter(date_format, data['cases_per_million'])

ax2 = ax.twinx()  
ax2.set_ylabel('Retail and Recreation Mobility')
recreation_scatter2 = ax2.scatter(date_format, data['retail_recreation'], color = "orange")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax2.tick_params(axis='y', labelcolor= "orange")

dates = data['date']
dates = dates[:43]
date_format = [pd.to_datetime(d) for d in dates]

data['date'] = data['date'].astype('datetime64')
data = data.groupby(data['date'].dt.date).mean()
x = data.iloc[:,1:8]
y = data.iloc[:,8:]

# Replaces NaN values present in the new columns by 0
x['fatalities_percent_change'] = x['fatalities_percent_change'].replace(np.NaN, 0)
x['cases_percent_change'] = x['cases_percent_change'].replace(np.NaN, 0)
y['cases_per_million'] = y['cases_per_million'].replace(np.NaN, 0)

# Fills the empty values taking mean values present in each column
x = x.replace("", np.NaN)
y = y.replace("", np.NaN)
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(x)
Y = imputer.fit_transform(y)

# Standardization using Gaussian Normal Distribution
X = preprocessing.scale(X)

# Train/Test Split (80%/20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Initalization of the Neural Network Model 
hidden_layers = 5
model = MLPRegressor(hidden_layers, validation_fraction = 0, solver='lbfgs').fit(X_train, Y_train)

# Obtaining predictions from the model
NN_predict = model.predict(X)

# Testing the model
NN_score = model.score(X_test, Y_test)
print('Score', NN_score)

# Plot the results
ax.grid()
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title=("Evolution of CoV-19 Cases per Million (19 different countries)"))
NN_plot, = ax.plot(date_format, NN_predict, color='green')
recreation_plot, = ax2.plot(date_format, data['retail_recreation'], color='red')
ax.legend((NN_plot,recreation_plot, cases_scatter2, recreation_scatter2), ('Perceptron\'s prediction', 'Average Retail and Recreation Mobility Change', 'Number of CoV-19 Cases per Million','Retail and Recreation Mobility'), loc='upper left', shadow=False)
ax.tick_params(axis='y', labelcolor= "blue")
plt.tight_layout()
plt.show()
```

    Score 0.9489916731019538
    


![svg](res/output_31_1.svg)


The graph above displays how the average number of confirmed cases per million evolves and confirms the previous conclusions.


## Conclusion
This project helped us by introducing ourselves to Machine Learning, demystifying such a relevant branch of Artificial Intelligence.
We believe that given the circumstances, the subject of the project was crucial to demonstrate one of the many practical uses of this field making it more engaging.
One of the main obstacles we had to face was the small dimension of our input dataset, mainly because of how recently this issue has surfaced. This has made it harder to implement useful methods such as K-Fold Cross Validation making the results somewhat unstable. Despite the difficulties, all of the algorithms were successfully implemented delivered good results. 


## References & Acknowledgements
    Google Community Mobility Reports and COVID Incidence. Dataset used with detailed information about it. Available at: https://www.kaggle.com/gustavomodelli/covid-community-measures​

    Supervised Learning Documentation. Official scikit-learn documentation about regression. Available at: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning​

    Different ways to compensate for missing values. Available at: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779​
    
    List of countries and dependencies by population. Available at: https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population
