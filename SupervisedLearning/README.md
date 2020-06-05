# *COVID-19 Community Mobility Reports:​ Regression*

### Authors:
- Ana Mafalda Santos, up201706791
- Diogo Silva, up201706892
- João Luz, up201703782

## Introduction
---
The main purpose of this project was to build a regression model using different **Supervised Learning** algorithms for the given dataset: **CoV-19 Community Mobility Report**.

After analyzing and handling our dataset, we aim to predict the number of people infected and the number of deaths caused by SARS-CoV-2 based on the variation of the population's mobility trends and some other factors. To make it less biased, we'll predict the confirmed cases and the number of deaths per million instead of using just the total amounts, as different countries have different population sizes.

## Description of the Problem/Dataset
---
Our dataset - **CoV-19 Community Mobility Report** - seeks to provide insights into what has changed due to policies aimed at combating COVID-19 and evaluate the changes in community activities and its relation to reduced confirmed cases of COVID-19. The reports chart movement trends, comparing to an expected baseline, over time by geography, across the following categories of places:

* **Retail & Recreation**: Mobility trends for places like restaurants, cafes, shopping centers, theme parks, museums, libraries, and movie theaters.
* **Grocery & Pharmacy**: Mobility trends for places like grocery markets, food warehouses, farmers' markets, specialty food shops, drug stores, and pharmacies.
* **Parks**: Mobility trends for places like national parks, public beaches, marinas, dog parks, plazas, and public gardens.
* **Transit stations**: Mobility trends for places like public transport hubs such as subway, bus, and train stations.
* **Workplaces**: Mobility trends for places of work.
* **Residential**: Mobility trends for places of residence.

The dataset also contains these CoV-19 statistics:

* **Total Cases**: Total number of people infected with the SARS-CoV-2.
* **Fatalities**: Total number of deaths caused by CoV-19.
* **Population**: Total number of inhabitants.
* **GDP per Capita (PPP)**: Gross Domestic Product (GDP) per capita based on Purchasing Power Parity (PPP), taking into account the relative cost of local goods, services and inflation rates of the country, rather than using international market exchange rates, which may distort the real differences in per capita income.
* **Health System Index**: Overall performance of the health system.
* **Human Development Index (HDI)**: Summary index based on life expectancy at birth, expected years of schooling for children and mean years of schooling for adults, and GNI per capita.
* **Government Response Stringency Index**: Additive score of nine indicators of government response to CoV-19: School closures, workplace closures, cancellation of public events, public information campaigns, stay at home policies, restrictions on internal movement, international travel controls, testing policy, and contact tracing.
* **Elderly Population (percentage)**: Percentage of the population above the age of 65 years old.

## Approach
---
We start by pre-processing the data we had. To compensate for missing and NaN values on the dataset, we used Median and Zero Imputation, to better infer those missing values from the existing part of the data. We also added new columns to extract more information from the existing columns:

* **New Cases (percentage)**: variation of the total cases of CoV-19 cases in comparison to the previous day.
* **New Fatalities (percentage)**: variation of the total fatalities of CoV-19 cases in comparison to the previous day.
* **Cases per Million**: total cases CoV-19 cases per million people.
* **Deaths per Million**: total CoV-19 deaths per million people.
* **Fatality Rate**: proportion of deaths from CoV-19 compared to the total number of confirmed CoV-19 cases.

The incubation period for COVID-19 (time between the exposure to the virus, becoming infected and symptoms onset) is on average 5-6 days, even though it can take up to 14 days. This means features such as the mobility changes and government-imposed policies (represented by the Stringency Index) are not immediately reflected in the evolution of the disease. Therefore, it makes sense to replace such features by the 7-10 previous days values when predicting a certain day.

In order to train and test the model, we partitioned the input data into training and testing data using the Train/Test Split.

To predict the number of people infected based on the available data, we used different algorithms with tweaked parameters for our regression model: **Support Vector Regression (SVR)**, **K-Nearest Neighbors**, **Decision Trees** and **Neural Networks (Multi-layer Perceptron)**.

After making the desired predictions, a limited sample of data is used in order to estimate how the models are expected to perform in general when used to make predictions on data not used during the training of these models.

At last, to reduce the dimension of our dataset by using Principal Component Analysis (PCA) which combines highly correlated variables together to form a smaller number of an artificial set of variables that account for the most variance in the data.

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
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_predict, cross_val_score
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

### A First Look At The Dataset

In the following graph, we can see how the mobility (in this case retail and recreation) varies once the number of cases starts to increase. The mobility along with other factors mentioned, will be used to predict the evolution of the number of cases.


```python
dates = data['date']
date_format = [pd.to_datetime(d) for d in dates]

_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
cases_scatter1 = ax.scatter(date_format, data['total_cases'])
ax.set(xlabel="Date", ylabel='Total Cases of CoV-19', title=("Evolution of CoV-19 Cases (19 different countries)"))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.tick_params(axis='y')
ax2 = ax.twinx()  
ax2.set_ylabel('Retail and Recreation Mobility')
retail_scatter1 = ax2.scatter(date_format, data['retail_recreation'], color = "orange")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax2.tick_params(axis='y')
ax.legend((cases_scatter1,retail_scatter1), ('Retail and Recreation Mobility', 'Number of CoV-19 Cases'), loc='upper left', shadow=False)
plt.show()
```


![svg](res/output_4_0.svg)



```python
# If we were taking the country column into account,
# it would intersting to take care of the Dummy Variable Trap.
# Alternatively, we chose to eliminate these columns completely.
print(data.head(100))
data = data.drop(columns = ['country','iso'])
```

    iso       country       date  grocery_pharmacy   parks  residential  \
    0   IN         India  2/23/2020            -1.192   0.717        0.969   
    1   IN         India  2/24/2020             0.036   2.103        0.189   
    2   IN         India  2/25/2020            -1.737   2.650        0.128   
    3   IN         India  2/26/2020            -0.197   1.497       -0.683   
    4   IN         India  2/27/2020            -0.699   3.362       -0.205   
    ..  ..           ...        ...               ...     ...          ...   
    95  IN         India  5/28/2020           -14.000 -54.000       19.000   
    96  IN         India  5/29/2020           -10.000 -54.000       19.000   
    97  ZA  South Africa  2/23/2020            -6.354 -20.793        2.833   
    98  ZA  South Africa  2/24/2020           -13.296 -10.143        0.614   
    99  ZA  South Africa  2/25/2020            -1.495  -4.721       -0.822   
    
        retail_recreation  transit_stations  workplaces  total_cases  fatalities  \
    0              -2.077            -0.484       2.213            3           0   
    1              -1.264             0.882       4.271            3           0   
    2              -0.691             1.087       1.055            3           0   
    3               0.358             2.629       3.250            3           0   
    4              -0.678             0.970       3.596            3           0   
    ..                ...               ...         ...          ...         ...   
    95            -69.000           -46.000     -40.000       158333        4531   
    96            -69.000           -45.000     -38.000       165799        4706   
    97             -9.238           -12.183      -0.602            0           0   
    98            -11.163            -2.237       4.742            0           0   
    99             -0.620             2.190       5.427            0           0   
    
        population  gdp_ppp  health_system_index  human_development_index  \
    0   1366417754     9027                0.617                    0.647   
    1   1366417754     9027                0.617                    0.647   
    2   1366417754     9027                0.617                    0.647   
    3   1366417754     9027                0.617                    0.647   
    4   1366417754     9027                0.617                    0.647   
    ..         ...      ...                  ...                      ...   
    95  1366417754     9027                0.617                    0.647   
    96  1366417754     9027                0.617                    0.647   
    97    58558270    13965                0.319                    0.705   
    98    58558270    13965                0.319                    0.705   
    99    58558270    13965                0.319                    0.705   
    
        response_stringency_index  age_above_65_percentage  
    0                      0.1019                      6.0  
    1                      0.1019                      6.0  
    2                      0.1019                      6.0  
    3                      0.1019                      6.0  
    4                      0.1019                      6.0  
    ..                        ...                      ...  
    95                     0.7917                      6.0  
    96                     0.7917                      6.0  
    97                     0.0278                      5.3  
    98                     0.0278                      5.3  
    99                     0.0278                      5.3  
    
    [100 rows x 17 columns]


### Extraction of new relevant columns

As mentioned before, we will be taking into account both the measures taken 7 and 10 days prior to each day of the dataset.


```python
data['fatalities_per_million'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    millions = data['population'][i]/1000000
    data['fatalities_per_million'][i]= data['fatalities'][i]/millions

data['fatalities_rate'] = pd.Series([])
for i in range(0,len(data['total_cases'])):
    if data['total_cases'][i] != 0:
        data['fatalities_rate'][i]= data['fatalities'][i] * 100 /data['total_cases'][i]
    else:
        data['fatalities_rate'][i] = 0

# Calculates daily percentual changes in number of fatalities
# data['cases_per_million'] = pd.Series([])
data.insert(13, "cases_per_million", 0) 
for i in range(0,len(data['total_cases'])):
    millions = data['population'][i]/1000000
    data['cases_per_million'][i]= data['total_cases'][i]/millions

column_names = ['date', 'grocery_pharmacy', 'parks', 'residential', 'retail_recreation',
       'transit_stations', 'workplaces','fatalities_rate','response_stringency_index','cases_per_million', 'fatalities_per_million','population']
# print(data.columns)
nentries= 97
for i in range(1, len(column_names) - 1):
    data[column_names[i] + '_previous_week'] = pd.Series([])
    data[column_names[i] + '_prev_10_days'] = pd.Series([])
    for j in range(0,len(data)):
            index = j % nentries
            if index >= 7:
                data[column_names[i] + '_previous_week'][j]= data[column_names[i]][j-7]
                if index >= 10:
                    data[column_names[i] + '_prev_10_days'][j] = data[column_names[i]][j-10]
                else:
                    data[column_names[i] + '_prev_10_days'][j] = 0
            else:
                data[column_names[i] + '_previous_week'][j]= 0
                data[column_names[i] + '_prev_10_days'][j] = 0
    if (i < len(column_names)-3):
        data = data.drop(columns = [column_names[i]])

data = data.drop(columns = ['total_cases'])
data = data.drop(columns = ['fatalities'])

column_order = ['date', 'grocery_pharmacy_previous_week', 'parks_previous_week', 'residential_previous_week', 'retail_recreation_previous_week',
       'transit_stations_previous_week', 'workplaces_previous_week', 'cases_per_million_previous_week','fatalities_per_million_previous_week', 'fatalities_rate_previous_week','gdp_ppp','health_system_index', 'human_development_index', 'response_stringency_index_previous_week', 'grocery_pharmacy_prev_10_days', 'parks_prev_10_days', 'residential_prev_10_days', 'retail_recreation_prev_10_days', 'transit_stations_prev_10_days', 'workplaces_prev_10_days', 'cases_per_million_prev_10_days','fatalities_per_million_prev_10_days', 'fatalities_rate_prev_10_days','response_stringency_index_prev_10_days','age_above_65_percentage', 'cases_per_million','fatalities_per_million']
data = data[column_order]

data.head(45)
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
      <th>grocery_pharmacy_previous_week</th>
      <th>parks_previous_week</th>
      <th>residential_previous_week</th>
      <th>retail_recreation_previous_week</th>
      <th>transit_stations_previous_week</th>
      <th>workplaces_previous_week</th>
      <th>cases_per_million_previous_week</th>
      <th>fatalities_per_million_previous_week</th>
      <th>fatalities_rate_previous_week</th>
      <th>...</th>
      <th>retail_recreation_prev_10_days</th>
      <th>transit_stations_prev_10_days</th>
      <th>workplaces_prev_10_days</th>
      <th>cases_per_million_prev_10_days</th>
      <th>fatalities_per_million_prev_10_days</th>
      <th>fatalities_rate_prev_10_days</th>
      <th>response_stringency_index_prev_10_days</th>
      <th>age_above_65_percentage</th>
      <th>cases_per_million</th>
      <th>fatalities_per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2/23/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2/24/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/25/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/26/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/27/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2/28/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2/29/2020</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3/1/2020</td>
      <td>-1.192</td>
      <td>0.717</td>
      <td>0.969</td>
      <td>-2.077</td>
      <td>-0.484</td>
      <td>2.213</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3/2/2020</td>
      <td>0.036</td>
      <td>2.103</td>
      <td>0.189</td>
      <td>-1.264</td>
      <td>0.882</td>
      <td>4.271</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3/3/2020</td>
      <td>-1.737</td>
      <td>2.650</td>
      <td>0.128</td>
      <td>-0.691</td>
      <td>1.087</td>
      <td>1.055</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3/4/2020</td>
      <td>-0.197</td>
      <td>1.497</td>
      <td>-0.683</td>
      <td>0.358</td>
      <td>2.629</td>
      <td>3.250</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-2.077</td>
      <td>-0.484</td>
      <td>2.213</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3/5/2020</td>
      <td>-0.699</td>
      <td>3.362</td>
      <td>-0.205</td>
      <td>-0.678</td>
      <td>0.970</td>
      <td>3.596</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-1.264</td>
      <td>0.882</td>
      <td>4.271</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3/6/2020</td>
      <td>1.423</td>
      <td>3.224</td>
      <td>0.039</td>
      <td>0.287</td>
      <td>1.490</td>
      <td>4.656</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.691</td>
      <td>1.087</td>
      <td>1.055</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3/7/2020</td>
      <td>-0.780</td>
      <td>-0.819</td>
      <td>0.320</td>
      <td>-1.767</td>
      <td>0.247</td>
      <td>7.707</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.358</td>
      <td>2.629</td>
      <td>3.250</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3/8/2020</td>
      <td>3.565</td>
      <td>0.657</td>
      <td>0.752</td>
      <td>0.967</td>
      <td>0.347</td>
      <td>3.381</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.678</td>
      <td>0.970</td>
      <td>3.596</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3/9/2020</td>
      <td>2.476</td>
      <td>2.754</td>
      <td>0.162</td>
      <td>-0.257</td>
      <td>-0.047</td>
      <td>4.848</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.287</td>
      <td>1.490</td>
      <td>4.656</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3/10/2020</td>
      <td>1.799</td>
      <td>2.918</td>
      <td>0.082</td>
      <td>-0.750</td>
      <td>0.340</td>
      <td>3.572</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-1.767</td>
      <td>0.247</td>
      <td>7.707</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3/11/2020</td>
      <td>5.344</td>
      <td>0.892</td>
      <td>0.001</td>
      <td>0.456</td>
      <td>2.705</td>
      <td>6.921</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.967</td>
      <td>0.347</td>
      <td>3.381</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3/12/2020</td>
      <td>2.983</td>
      <td>2.627</td>
      <td>0.362</td>
      <td>-0.729</td>
      <td>1.772</td>
      <td>4.969</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.257</td>
      <td>-0.047</td>
      <td>4.848</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.000732</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3/13/2020</td>
      <td>2.058</td>
      <td>-0.322</td>
      <td>1.008</td>
      <td>-1.308</td>
      <td>2.624</td>
      <td>4.825</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.750</td>
      <td>0.340</td>
      <td>3.572</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1019</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3/14/2020</td>
      <td>5.812</td>
      <td>1.780</td>
      <td>0.112</td>
      <td>0.240</td>
      <td>4.909</td>
      <td>6.876</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.456</td>
      <td>2.705</td>
      <td>6.921</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1574</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3/15/2020</td>
      <td>8.745</td>
      <td>1.710</td>
      <td>0.200</td>
      <td>2.160</td>
      <td>3.201</td>
      <td>6.030</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-0.729</td>
      <td>1.772</td>
      <td>4.969</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3/16/2020</td>
      <td>5.905</td>
      <td>4.058</td>
      <td>3.498</td>
      <td>-1.361</td>
      <td>-6.067</td>
      <td>-12.394</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-1.308</td>
      <td>2.624</td>
      <td>4.825</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.001464</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3/17/2020</td>
      <td>-22.915</td>
      <td>-3.624</td>
      <td>11.229</td>
      <td>-19.005</td>
      <td>-22.172</td>
      <td>-43.880</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.240</td>
      <td>4.909</td>
      <td>6.876</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.002196</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3/18/2020</td>
      <td>-1.171</td>
      <td>-1.798</td>
      <td>0.727</td>
      <td>-5.180</td>
      <td>-0.845</td>
      <td>-5.349</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.160</td>
      <td>3.201</td>
      <td>6.030</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.002196</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3/19/2020</td>
      <td>0.572</td>
      <td>1.889</td>
      <td>0.105</td>
      <td>-3.315</td>
      <td>-0.460</td>
      <td>-0.364</td>
      <td>0.0</td>
      <td>0.000732</td>
      <td>1.351351</td>
      <td>...</td>
      <td>-1.361</td>
      <td>-6.067</td>
      <td>-12.394</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.002927</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3/20/2020</td>
      <td>1.433</td>
      <td>-0.126</td>
      <td>1.107</td>
      <td>-4.978</td>
      <td>-1.971</td>
      <td>0.354</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>2.469136</td>
      <td>...</td>
      <td>-19.005</td>
      <td>-22.172</td>
      <td>-43.880</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.002927</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3/21/2020</td>
      <td>-2.517</td>
      <td>-5.592</td>
      <td>3.640</td>
      <td>-10.598</td>
      <td>-7.764</td>
      <td>-5.534</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>2.380952</td>
      <td>...</td>
      <td>-5.180</td>
      <td>-0.845</td>
      <td>-5.349</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.002927</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3/22/2020</td>
      <td>-0.894</td>
      <td>-6.498</td>
      <td>2.600</td>
      <td>-11.093</td>
      <td>-6.608</td>
      <td>1.664</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>1.818182</td>
      <td>...</td>
      <td>-3.315</td>
      <td>-0.460</td>
      <td>-0.364</td>
      <td>0.0</td>
      <td>0.000732</td>
      <td>1.351351</td>
      <td>0.2685</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.005123</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3/23/2020</td>
      <td>1.561</td>
      <td>-1.478</td>
      <td>2.714</td>
      <td>-7.599</td>
      <td>-5.834</td>
      <td>-3.938</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>1.754386</td>
      <td>...</td>
      <td>-4.978</td>
      <td>-1.971</td>
      <td>0.354</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>2.469136</td>
      <td>0.3333</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.006587</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3/24/2020</td>
      <td>0.272</td>
      <td>-2.160</td>
      <td>3.142</td>
      <td>-10.512</td>
      <td>-8.524</td>
      <td>-8.356</td>
      <td>0.0</td>
      <td>0.002196</td>
      <td>2.189781</td>
      <td>...</td>
      <td>-10.598</td>
      <td>-7.764</td>
      <td>-5.534</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>2.380952</td>
      <td>0.3611</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.007318</td>
    </tr>
    <tr>
      <th>31</th>
      <td>3/25/2020</td>
      <td>1.449</td>
      <td>-5.868</td>
      <td>3.858</td>
      <td>-11.895</td>
      <td>-9.969</td>
      <td>-7.610</td>
      <td>0.0</td>
      <td>0.002196</td>
      <td>1.986755</td>
      <td>...</td>
      <td>-11.093</td>
      <td>-6.608</td>
      <td>1.664</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>1.818182</td>
      <td>0.3889</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.007318</td>
    </tr>
    <tr>
      <th>32</th>
      <td>3/26/2020</td>
      <td>1.126</td>
      <td>-6.891</td>
      <td>4.914</td>
      <td>-15.786</td>
      <td>-14.009</td>
      <td>-11.143</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>2.312139</td>
      <td>...</td>
      <td>-7.599</td>
      <td>-5.834</td>
      <td>-3.938</td>
      <td>0.0</td>
      <td>0.001464</td>
      <td>1.754386</td>
      <td>0.4815</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.011709</td>
    </tr>
    <tr>
      <th>33</th>
      <td>3/27/2020</td>
      <td>0.914</td>
      <td>-12.683</td>
      <td>6.984</td>
      <td>-22.318</td>
      <td>-18.195</td>
      <td>-14.866</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>1.793722</td>
      <td>...</td>
      <td>-10.512</td>
      <td>-8.524</td>
      <td>-8.356</td>
      <td>0.0</td>
      <td>0.002196</td>
      <td>2.189781</td>
      <td>0.4815</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.013905</td>
    </tr>
    <tr>
      <th>34</th>
      <td>3/28/2020</td>
      <td>-4.040</td>
      <td>-22.680</td>
      <td>8.271</td>
      <td>-33.737</td>
      <td>-25.235</td>
      <td>-17.765</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>1.269841</td>
      <td>...</td>
      <td>-11.895</td>
      <td>-9.969</td>
      <td>-7.610</td>
      <td>0.0</td>
      <td>0.002196</td>
      <td>1.986755</td>
      <td>0.5000</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.013905</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3/29/2020</td>
      <td>-76.363</td>
      <td>-57.624</td>
      <td>21.371</td>
      <td>-78.074</td>
      <td>-69.511</td>
      <td>-49.251</td>
      <td>0.0</td>
      <td>0.005123</td>
      <td>1.944444</td>
      <td>...</td>
      <td>-15.786</td>
      <td>-14.009</td>
      <td>-11.143</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>2.312139</td>
      <td>0.5972</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.019760</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3/30/2020</td>
      <td>-27.006</td>
      <td>-35.079</td>
      <td>19.905</td>
      <td>-54.169</td>
      <td>-52.194</td>
      <td>-50.413</td>
      <td>0.0</td>
      <td>0.006587</td>
      <td>1.923077</td>
      <td>...</td>
      <td>-22.318</td>
      <td>-18.195</td>
      <td>-14.866</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>1.793722</td>
      <td>0.7083</td>
      <td>6.0</td>
      <td>0</td>
      <td>0.023419</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3/31/2020</td>
      <td>-47.760</td>
      <td>-44.487</td>
      <td>25.194</td>
      <td>-67.533</td>
      <td>-63.670</td>
      <td>-64.912</td>
      <td>0.0</td>
      <td>0.007318</td>
      <td>1.926782</td>
      <td>...</td>
      <td>-33.737</td>
      <td>-25.235</td>
      <td>-17.765</td>
      <td>0.0</td>
      <td>0.002927</td>
      <td>1.269841</td>
      <td>0.7639</td>
      <td>6.0</td>
      <td>1</td>
      <td>0.025614</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4/1/2020</td>
      <td>-64.123</td>
      <td>-52.799</td>
      <td>30.267</td>
      <td>-76.904</td>
      <td>-73.230</td>
      <td>-71.834</td>
      <td>0.0</td>
      <td>0.007318</td>
      <td>1.650165</td>
      <td>...</td>
      <td>-78.074</td>
      <td>-69.511</td>
      <td>-49.251</td>
      <td>0.0</td>
      <td>0.005123</td>
      <td>1.944444</td>
      <td>1.0000</td>
      <td>6.0</td>
      <td>1</td>
      <td>0.030005</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4/2/2020</td>
      <td>-63.799</td>
      <td>-50.977</td>
      <td>29.981</td>
      <td>-76.438</td>
      <td>-73.147</td>
      <td>-70.256</td>
      <td>0.0</td>
      <td>0.011709</td>
      <td>2.305476</td>
      <td>...</td>
      <td>-54.169</td>
      <td>-52.194</td>
      <td>-50.413</td>
      <td>0.0</td>
      <td>0.006587</td>
      <td>1.923077</td>
      <td>0.8657</td>
      <td>6.0</td>
      <td>1</td>
      <td>0.038788</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4/3/2020</td>
      <td>-63.591</td>
      <td>-51.872</td>
      <td>30.885</td>
      <td>-76.733</td>
      <td>-73.294</td>
      <td>-69.780</td>
      <td>0.0</td>
      <td>0.013905</td>
      <td>2.278177</td>
      <td>...</td>
      <td>-67.533</td>
      <td>-63.670</td>
      <td>-64.912</td>
      <td>0.0</td>
      <td>0.007318</td>
      <td>1.926782</td>
      <td>0.8657</td>
      <td>6.0</td>
      <td>1</td>
      <td>0.045374</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4/4/2020</td>
      <td>-65.200</td>
      <td>-54.355</td>
      <td>29.099</td>
      <td>-77.923</td>
      <td>-73.296</td>
      <td>-65.786</td>
      <td>0.0</td>
      <td>0.013905</td>
      <td>2.069717</td>
      <td>...</td>
      <td>-76.904</td>
      <td>-73.230</td>
      <td>-71.834</td>
      <td>0.0</td>
      <td>0.007318</td>
      <td>1.650165</td>
      <td>1.0000</td>
      <td>6.0</td>
      <td>2</td>
      <td>0.054888</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4/5/2020</td>
      <td>-65.287</td>
      <td>-56.524</td>
      <td>21.993</td>
      <td>-77.059</td>
      <td>-71.299</td>
      <td>-46.674</td>
      <td>0.0</td>
      <td>0.019760</td>
      <td>2.636719</td>
      <td>...</td>
      <td>-76.438</td>
      <td>-73.147</td>
      <td>-70.256</td>
      <td>0.0</td>
      <td>0.011709</td>
      <td>2.305476</td>
      <td>1.0000</td>
      <td>6.0</td>
      <td>2</td>
      <td>0.060743</td>
    </tr>
    <tr>
      <th>43</th>
      <td>4/6/2020</td>
      <td>-63.416</td>
      <td>-51.189</td>
      <td>30.121</td>
      <td>-75.924</td>
      <td>-73.761</td>
      <td>-67.977</td>
      <td>0.0</td>
      <td>0.023419</td>
      <td>2.557954</td>
      <td>...</td>
      <td>-76.733</td>
      <td>-73.294</td>
      <td>-69.780</td>
      <td>0.0</td>
      <td>0.013905</td>
      <td>2.278177</td>
      <td>1.0000</td>
      <td>6.0</td>
      <td>3</td>
      <td>0.081234</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4/7/2020</td>
      <td>-63.001</td>
      <td>-50.224</td>
      <td>28.717</td>
      <td>-75.472</td>
      <td>-72.720</td>
      <td>-68.224</td>
      <td>1.0</td>
      <td>0.025614</td>
      <td>2.505369</td>
      <td>...</td>
      <td>-77.923</td>
      <td>-73.296</td>
      <td>-65.786</td>
      <td>0.0</td>
      <td>0.013905</td>
      <td>2.069717</td>
      <td>1.0000</td>
      <td>6.0</td>
      <td>3</td>
      <td>0.090748</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 27 columns</p>
</div>



We will start by using data from every coutry in the dataset to predict the number of cases of SARS-CoV-2 per million in Italy. First, need to define the Total Number of Cases per Million as our **target variable** and isolate it from our **independent variables**. After that, we can start pre-processing our data so it is ready to be used by our models. 


```python
X = data.iloc[:,1:25]
Y = data.iloc[:,25:26]
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
      <th>grocery_pharmacy_previous_week</th>
      <th>parks_previous_week</th>
      <th>residential_previous_week</th>
      <th>retail_recreation_previous_week</th>
      <th>transit_stations_previous_week</th>
      <th>workplaces_previous_week</th>
      <th>cases_per_million_previous_week</th>
      <th>fatalities_per_million_previous_week</th>
      <th>fatalities_rate_previous_week</th>
      <th>gdp_ppp</th>
      <th>...</th>
      <th>parks_prev_10_days</th>
      <th>residential_prev_10_days</th>
      <th>retail_recreation_prev_10_days</th>
      <th>transit_stations_prev_10_days</th>
      <th>workplaces_prev_10_days</th>
      <th>cases_per_million_prev_10_days</th>
      <th>fatalities_per_million_prev_10_days</th>
      <th>fatalities_rate_prev_10_days</th>
      <th>response_stringency_index_prev_10_days</th>
      <th>age_above_65_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9027</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1838</th>
      <td>-8.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>-26.0</td>
      <td>-43.0</td>
      <td>-40.0</td>
      <td>4553.0</td>
      <td>255.970769</td>
      <td>5.621906</td>
      <td>67426</td>
      <td>...</td>
      <td>21.0</td>
      <td>16.0</td>
      <td>-29.0</td>
      <td>-40.0</td>
      <td>-42.0</td>
      <td>4354.0</td>
      <td>247.437499</td>
      <td>5.682396</td>
      <td>0.7269</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>1839</th>
      <td>-4.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>-24.0</td>
      <td>-40.0</td>
      <td>-41.0</td>
      <td>4612.0</td>
      <td>260.301222</td>
      <td>5.642963</td>
      <td>67426</td>
      <td>...</td>
      <td>32.0</td>
      <td>9.0</td>
      <td>-30.0</td>
      <td>-34.0</td>
      <td>-24.0</td>
      <td>4428.0</td>
      <td>251.178402</td>
      <td>5.671231</td>
      <td>0.7269</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>1840</th>
      <td>-3.0</td>
      <td>23.0</td>
      <td>15.0</td>
      <td>-22.0</td>
      <td>-38.0</td>
      <td>-41.0</td>
      <td>4680.0</td>
      <td>264.944683</td>
      <td>5.660218</td>
      <td>67426</td>
      <td>...</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>-30.0</td>
      <td>-39.0</td>
      <td>-25.0</td>
      <td>4488.0</td>
      <td>253.563950</td>
      <td>5.648621</td>
      <td>0.7269</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>1841</th>
      <td>-1.0</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>-22.0</td>
      <td>-37.0</td>
      <td>-41.0</td>
      <td>4756.0</td>
      <td>268.852726</td>
      <td>5.651912</td>
      <td>67426</td>
      <td>...</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>-26.0</td>
      <td>-43.0</td>
      <td>-40.0</td>
      <td>4553.0</td>
      <td>255.970769</td>
      <td>5.621906</td>
      <td>0.7269</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>1842</th>
      <td>-2.0</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>-24.0</td>
      <td>-36.0</td>
      <td>-41.0</td>
      <td>4829.0</td>
      <td>272.687836</td>
      <td>5.646242</td>
      <td>67426</td>
      <td>...</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>-24.0</td>
      <td>-40.0</td>
      <td>-41.0</td>
      <td>4612.0</td>
      <td>260.301222</td>
      <td>5.642963</td>
      <td>0.7269</td>
      <td>15.4</td>
    </tr>
  </tbody>
</table>
<p>1843 rows × 24 columns</p>
</div>




```python
# Fills the empty values taking mean values present in each column (Mean Inputation)
X = X.replace("", np.NaN)
Y = Y.replace("", np.NaN)
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

# print(Y)
```

To prevent a feature that has a variance that is orders of magnitude larger than others, from dominating the objective function and make the estimator unable to learn from other features correctly as expected, it is wise to perform standardization of such features.


```python
# Standardization using Gaussian Normal Distribution
X = preprocessing.scale(X)
```

Once pre-processed, our data can now be divided into subsets meant for training and testing our models. We decided to apply Train/Test Split (70%/30%) which randomly selects the input and output data as training or testing data.


```python
# Create training and testing vars
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
```

We are now ready to start training and testing our models.

### Support Vector Regression

We start by using the SVR algorithm, which is an extension of the Support Vector Machine algorithm, usually used in Classification models, but applied to Regression. The **scikit-learn** library provides several alternatives for kernel functions, namely **Radial Basis** (the default option according to the documentation), **Linear**, or **Polynomial**. Besides being relatively memory efficient, the SVR algorithm works well with high dimensional spaces and datasets without too much noise. Making it a good contender for this problem.
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
X_pred = X[1067:1163,:]
Y_pred_label = Y[1067:1163,:]

linear_predictions = lm.predict(X_pred)
svr_lin_predictions = svr_lin_model.predict(X_pred)
svr_rbf_predictions = svr_rbf_model.predict(X_pred)
svr_poly_predictions = svr_poly_model.predict(X_pred)

# Testing models
linear_score = model.score(X_test, Y_test)
svr_lin_score = svr_lin_model.score(X_test, Y_test)
svr_rbf_score = svr_rbf_model.score(X_test, Y_test)
svr_poly_score = svr_poly_model.score(X_test, Y_test)

print("Linear Regression Score:", linear_score)
print("Linear SVR Score: %s" % (svr_lin_score))
print("RBF SVR Score: %s" % (svr_rbf_score))
print("Polynomial SVR Score: %s" % (svr_poly_score))

# Evaluating the models
linear_eval = cross_val_score(model, X, Y, cv=10)
svr_lin_eval = cross_val_score(svr_lin, X, Y, cv=10)
svr_rbf_eval = cross_val_score(svr_rbf, X, Y, cv=10)
svr_poly_eval = cross_val_score(svr_poly, X, Y, cv=10)

print("\nLinear Regression CV:", np.mean(linear_eval))
print("Linear SVR CV: %s" % (np.mean(svr_lin_eval)))
print("RBF SVR CV: %s" % (np.mean (svr_rbf_eval)))
print("Polynomial SVR CV: %s" % (np.mean(svr_poly_eval)))
```

    Linear Regression Score: 0.9979494304375061
    Linear SVR Score: 0.9947908879500098
    RBF SVR Score: 0.933775210401122
    Polynomial SVR Score: 0.9955054633947337
    
    Linear Regression CV: 0.9352300773824235
    Linear SVR CV: 0.8559185630824956
    RBF SVR CV: -6.914304568700567
    Polynomial SVR CV: 0.3800585713300326



```python
# Visual Comparison of the predictions obtained by different 
# parameterizations of the SVR model compared to a Linear Regression model
dates = data['date']
date_format = [pd.to_datetime(d) for d in dates]

_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
ax.scatter(date_format[0:96], Y_pred_label)
linear_plot, = ax.plot(date_format[0:96], linear_predictions)
svr_poly_plot, = ax.plot(date_format[0:96], svr_poly_predictions)
svr_lin_plot, = ax.plot(date_format[0:96], svr_lin_predictions)
svr_rbf_plot, = ax.plot(date_format[0:96], svr_rbf_predictions)
ax.legend((linear_plot, svr_poly_plot, svr_lin_plot, svr_rbf_plot), ('Linear Regression', 'SVR Polynomial Kernel','SVR Linear Kernel', 'SVR Radial Basis Function Kernel'), loc='upper left', shadow=False)

ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title=("Evolution of CoV-19 Cases per Million"))
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.tick_params(axis='y', labelcolor= "blue")
```


![svg](res/output_17_0.svg)


As we can see by the scores obtained by the comparison of the predictions with the labeled data in several experiments, the performance of the SVR algorithms tends to be better with the exception of the Radial Basis kernel, which is slightly less accurate.
### K-Nearest Neighbor

We proceed to apply the K-Nearest Neighbor algorithm. Once again, we are given two alternatives for the weights specified for the "neighbors". Using a uniform weight distribution, each of the k-neighbor's weight is taken into account equally. On the other hand, when considering the distance between these neighbors, the weight decreases with the distance.


```python
n_neighbors = 3

# K-Nearest Neighbor Models
knn_uniform = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn_distance = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')

# Fitting/Training and obtaining predictions from models
knn_predictions_uniform = knn_uniform.fit(X_train, Y_train).predict(X_pred)
knn_predictions_distance = knn_distance.fit(X_train, Y_train).predict(X_pred)

# Testing models
knn_uniform_score = knn_uniform.score(X_test, Y_test)
knn_distance_score = knn_distance.score(X_test, Y_test)

print("KNN Regression Model - Uniform:  %s"  % (knn_uniform_score))
print("KNN Regression Model - Distance:  %s"  % (knn_distance_score))

# Evaluating the models
uniform_eval = cross_val_score(knn_uniform, X, Y, cv=10)
distance_eval = cross_val_score(knn_distance, X, Y, cv=10)

print("\nKNN Regression Model - Uniform CV:", np.mean(uniform_eval))
print("KNN Regression Model - Uniform CV: %s" % (np.mean(distance_eval)))

# Visual Comparison of the predictions obtained by different 
# parameterizations of the KNN model
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[0:96], Y_pred_label, color='darkorange', label='data')
unif, = ax.plot(date_format[0:96], knn_predictions_uniform, color='navy')
dist, = ax.plot(date_format[0:96], knn_predictions_distance, color='darkgreen')
ax.legend((unif, dist, labeled), ('Uniform', 'Distance','Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million: K-Nearest Neighbor")
ax.axis('tight')
plt.tight_layout()
plt.show()
```

    KNN Regression Model - Uniform:  0.9917764223373067
    KNN Regression Model - Distance:  0.9928581307644214
    
    KNN Regression Model - Uniform CV: -0.7866985402520565
    KNN Regression Model - Uniform CV: -0.7772326125497141



![svg](res/output_19_1.svg)


The results obtained with the parameterizations of this algorithm were very promising, with a slight advantage for the distance-based approach. However, according to the Cross Validation results, it doesn't seem to be the most adequate of the algorithms.

### Decision Tree Regression

Another worthy approach for this regression problem is the Decision Tree algorithm. It's important to observe the influence of the *max_depth* parameter of the model to see how it affects its performance. With that in mind, we chose to compare two DT models with *max_depth* values of 5 and 10.

Usually, it would be risky to use a high value for the *max_depth* parameter of a decision tree since it can lead to the model learning from noise present in the data and be prone to **overfit**. 


```python
# Decision Tree Models
dt_regressor_5 = DecisionTreeRegressor(max_depth=5)
dt_regressor_10 = DecisionTreeRegressor(max_depth=10)

# Fitting/Training models
dt_regressor_5.fit(X_train, Y_train)
dt_regressor_10.fit(X_train, Y_train)

# Obtaining predictions from models
X_pred = X[1746:1842,:]
Y_pred_label = Y[1746:1842,:]
max_depth5_prediction = dt_regressor_5.predict(X_pred)
max_depth10_prediction = dt_regressor_10.predict(X_pred)

# Testing models
dt_regressor_5_score = dt_regressor_5.score(X_test,Y_test)
dt_regressor_10_score = dt_regressor_10.score(X_test,Y_test)

print("DT5", dt_regressor_5_score)
print("DT10", dt_regressor_10_score)

# Evaluating the models
dt_regressor_5_eval = cross_val_score(dt_regressor_5, X, Y, cv=10)
dt_regressor_10_eval = cross_val_score(dt_regressor_10, X, Y, cv=10)

print("\nDT5 Regression Model CV:", np.mean(dt_regressor_5_eval))
print("DT10 Regression Model CV: %s" % (np.mean(dt_regressor_10_eval)))

# Plot the results
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[1746:1842], Y_pred_label, color='darkorange', label='data')
max_depth5, = ax.plot(date_format[1746:1842], max_depth5_prediction, color='cornflowerblue')
max_depth10, = ax.plot(date_format[1746:1842], max_depth10_prediction, color='yellowgreen')
ax.legend((max_depth5, max_depth10, labeled), ('max_depth-2', 'max_depth-10','Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million: Decision Tree Regression")
plt.tight_layout()
plt.show()
```

    DT5 0.9886262023570297
    DT10 0.995054211341978
    
    DT5 Regression Model CV: 0.8075855905519778
    DT10 Regression Model CV: 0.6761809315858694



![svg](res/output_21_1.svg)


Fortunately, given that our data does not contain noise and was handled carefully beforehand, the use of a high *max_depth* parameter tends to be beneficial.

### Neural Network - Multi-layer Perceptron

Finally, we continue our study by using a Neural Network. For this, we are using a multi-layer perceptron which iteratively corrects its parameters according to the partial derivatives of the loss function.
The default solver *adam* works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, *lbfgs* can converge faster and perform better.


```python
hidden_layer_neurons = (150,150,150)

# Fitting/Training Neural Network (MLP) Model
model = MLPRegressor(hidden_layer_neurons, validation_fraction = 0, solver='lbfgs').fit(X_train, Y_train)

# Obtaining predictions from models
X_pred = X[1746:,:]
Y_pred_label = Y[1746:,:]
print(X_pred.shape)
NN_predict = model.predict(X_pred)

# Testing models
NN_score = model.score(X_test, Y_test)
print('Score', NN_score)

# Evaluating the models
neural_network_eval = cross_val_score(model, X, Y, cv=10)
print("\nNeural Network CV:", np.mean(neural_network_eval))

# Visual Comparison of the predictions obtained by the NN model
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[0:97], Y_pred_label, color='darkorange', label='data')
NN_plot, = ax.plot(date_format[0:97], NN_predict, color='cornflowerblue')
ax.legend((NN_plot,labeled), ('Perceptron\'s prediction', 'Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million : Neural network - Multi-layer Perceptron")
plt.tight_layout()
plt.show()
```

    Score 0.9992549099215091
    
    Neural Network CV: -0.009368721472515707



![svg](res/output_23_1.svg)


Although it requires a lot of computational power, it provides good and consistent results.

## Score Comparison
---

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

ax.bar('Decision Tree (Depth = 2)', dt_regressor_5_score, width=0.8, bottom=None, align='center', data=None)
ax.bar('Decision Tree (Depth = 10)', dt_regressor_10_score, width=0.8, bottom=None, align='center', data=None)

ax.bar('Neural Network (MLP)', NN_score, width=0.8, bottom=None, align='center', data=None)

plt.xticks(rotation=90)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8], <a list of 9 Text major ticklabel objects>)




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

    [0.52695139 0.21804284 0.08574569 0.03770935 0.0317162  0.02001957
     0.01883774 0.01365975]
    [[-0.22705438 -0.17155543  0.25834322 -0.26500695 -0.26273942 -0.2553553
       0.15791346  0.15620307  0.1744258  -0.01939889  0.02695976 -0.01675665
       0.24668256 -0.22913987 -0.18633287  0.26155721 -0.26931231 -0.26689638
      -0.25879737  0.15318271  0.15113959  0.17371857  0.25080096  0.00683449]
     [ 0.08948724  0.21257465 -0.08534935  0.07292442  0.05866224  0.02906397
       0.29400374  0.30255723  0.21383759  0.31191134  0.32243211  0.33381656
      -0.07289314  0.0717909   0.19189409 -0.07158627  0.05706881  0.04562528
       0.02166508  0.29543244  0.30154599  0.21567982 -0.06182019  0.32672979]
     [ 0.03023318  0.02204866 -0.10504438  0.07228996  0.13956892  0.1083392
       0.21664636  0.29031138  0.21813498 -0.41248338 -0.33586724 -0.41274299
      -0.09356686 -0.01757755 -0.0070349  -0.07886548  0.04428932  0.11098192
       0.08353434  0.23150652  0.30144478  0.23390742 -0.08142281 -0.27680566]
     [ 0.14241524  0.521042    0.09887857  0.0214577  -0.09546736 -0.15940208
       0.13466449 -0.0657137  -0.14245207  0.09103277 -0.32448361 -0.0267124
       0.2460455   0.14561307  0.50097254  0.08463738  0.01841075 -0.09448974
      -0.13166386  0.13574243 -0.0601997  -0.12966307  0.24618023 -0.20121816]
     [ 0.07053279  0.16765659  0.03574879 -0.01039014 -0.05191005 -0.05976522
      -0.35615174 -0.12128708  0.53051086 -0.14987574  0.02573697 -0.05766964
       0.0792091   0.14104084  0.19743445 -0.00084746  0.02881229 -0.01585696
      -0.01539491 -0.36224689 -0.1380283   0.51301018  0.06959505  0.15828079]
     [-0.3596583  -0.27876458  0.15815423 -0.23131052 -0.21085228 -0.13150811
       0.0549782   0.04582021 -0.03058082 -0.0025971   0.02255709 -0.00638519
       0.14838542  0.64110124  0.10718714 -0.2169156   0.19833568  0.19207793
       0.25986868  0.05757491  0.05373862 -0.00904725  0.04781473 -0.0463488 ]
     [-0.54948022  0.22642379  0.10986629 -0.08728031 -0.00467984 -0.25255769
      -0.04116985  0.02891458 -0.08480377 -0.17866335 -0.07311712 -0.08713897
      -0.35967438 -0.20323284  0.28317215  0.01918668  0.02065501  0.07966335
      -0.14330982 -0.04932605  0.0141674  -0.10908099 -0.34858169  0.30683768]
     [-0.20973279  0.02475815  0.03220875  0.07789454 -0.06809816 -0.09489423
      -0.0535662  -0.00632409  0.15899517  0.44059457  0.14840677  0.03674429
      -0.23752449 -0.07721261  0.03773092  0.01879703  0.1001582  -0.04424842
      -0.07435745 -0.07060101 -0.01994816  0.15770558 -0.2231597  -0.72611011]]


Using this analysis, we obtain the principal components that make up for at least 75% of the variance ratio. These are the **Health System Index** and **Human Development Index**.

### Another Approach
---
Knowing that different countries apply different policies, have different lifestyles, and social and economical disparities, we believe it would be interesting to see if the model could correctly predict the evolution of the total number of cases per million of a given country using only data of similar countries in terms of GDP, HDI and Government Stringency, since these factors can directly affect the rate at which the virus propagates.

Having this in mind, Mexico, Argentina and South Africa seemed like good candidates, so the first two were used to predict the evolution of the third.


```python
mexicoX = data.iloc[485:581,1:25]
mexicoY = data.iloc[485:581,25:26]

argentinaX = data.iloc[388:484, 1:25]
argentinaY = data.iloc[388:484, 25:26]

X = mexicoX.append(argentinaX)
Y = mexicoY.append(argentinaY)

safricaX = data.iloc[97:193, 1:25]
safricaY = data.iloc[97:193, 25:26]

# Fills the empty values taking mean values present in each column (Mean Inputation)
X = X.replace("", np.NaN)
Y = Y.replace("", np.NaN)
safricaX = safricaX.replace("", np.NaN)
safricaY = safricaY.replace("", np.NaN)
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)
safricaX =  imputer.fit_transform(safricaX)
safricaY =  imputer.fit_transform(safricaY)

# Standardization using Gaussian Normal Distribution
X = preprocessing.scale(X)
safricaX = preprocessing.scale(safricaX)

# Initalization of the Neural Network Model 
hidden_layers = (150,150,150)
model = MLPRegressor(hidden_layers, validation_fraction = 0, solver='lbfgs').fit(X, Y)

# Obtaining predictions from the model
NN_predict = model.predict(safricaX)

# Testing the model
NN_score = model.score(safricaX, safricaY)
print('Score', NN_score)

# Plot the results
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[0:96], safricaY, color='purple', label='data')

NN_plot, = ax.plot(date_format[0:96], NN_predict, color='cornflowerblue')
ax.legend((NN_plot,labeled), ('Perceptron\'s prediction', 'Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Cases per Million', title = "Evolution of CoV-19 Cases per Million : Neural network - Multi-layer Perceptron")
plt.tight_layout()
plt.show()
```

    Score 0.9633940916786529



![svg](res/output_29_1.svg)


As seen in the graph, the model correctly predicts most of the evolution of South Africa's CoV-19 cases.

### Confirmed Deaths
---

Since CoV-19 is more likely to cause the death of elderly people, we decided to pick countries with similar elderly population proportions, comparable GDP per capita and identical health system performances.

Such countries meeting this criteria were the United Kingdom, Italy, Spain, Canada and France. Using the first four countries we used the model to predict the number of confirmed deaths per million in France.


```python
ukX = data.iloc[1067:1163,1:26]
ukY = data.iloc[1067:1163,26:]

germanyX = data.iloc[1455:1551, 1:26]
germanyY = data.iloc[1455:1551, 26:]

X = ukX.append(germanyX)
Y = ukY.append(germanyY)

italyX = data.iloc[679:775, 1:26]
italyY = data.iloc[679:775, 26:]

X = X.append(italyX)
Y = Y.append(italyY)

spainX = data.iloc[776:872, 1:26]
spainY = data.iloc[776:872, 26:]

X = X.append(spainX)
Y = Y.append(spainY)

canadaX = data.iloc[1261:1357, 1:26]
canadaY = data.iloc[1261:1357, 26:]

X = X.append(canadaX)
Y = Y.append(canadaY)

franceX = data.iloc[1164:1260, 1:26]
franceY = data.iloc[1164:1260, 26:]

# Fills the empty values taking mean values present in each column (Mean Inputation)
X = X.replace("", np.NaN)
Y = Y.replace("", np.NaN)
franceX = franceX.replace("", np.NaN)
franceY = franceY.replace("", np.NaN)
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)
franceX =  imputer.fit_transform(franceX)
franceY =  imputer.fit_transform(franceY)

# Standardization using Gaussian Normal Distribution
X = preprocessing.scale(X)
franceX = preprocessing.scale(franceX)

# Initalization of the Neural Network Model 
hidden_layers = (150,150,150)
model = MLPRegressor(hidden_layers, validation_fraction = 0, solver='lbfgs').fit(X, Y)

# Obtaining predictions from the model
NN_predict = model.predict(franceX)

# Testing the model
NN_score = model.score(franceX, franceY)
print('Score', NN_score)

# Plot the results
_, ax = plt.subplots(figsize=(24, 10))
ax.grid()
labeled = ax.scatter(date_format[0:96], franceY, color='purple', label='data')

NN_plot, = ax.plot(date_format[0:96], NN_predict, color='cornflowerblue')
ax.legend((NN_plot,labeled), ('Perceptron\'s prediction', 'Labeled Data'), loc='upper left', shadow=False)
ax.set(xlabel="Date", ylabel='Number of CoV-19 Fatalities per Million', title = "Evolution of CoV-19 Cases per Million : Neural network - Multi-layer Perceptron")
plt.tight_layout()
plt.show()
```

    Score 0.9829962133191966



![svg](res/output_31_1.svg)


The graph above displays how the average number of confirmed deaths per million evolves and confirms the previous conclusions.

## Conclusion
This project helped us by introducing ourselves to Machine Learning, demystifying such a relevant branch of Artificial Intelligence.
We believe that given the circumstances, the subject of the project was crucial to demonstrate one of the many practical uses of this field making it more engaging.
Once the learning curve of both the programming language and the concepts of Machine Learning was passed, all of the algorithms were successfully implemented and delivered good results.

## References & Acknowledgements
    Google Community Mobility Reports and COVID Incidence. Dataset used with detailed information about it. Available at: https://www.kaggle.com/gustavomodelli/covid-community-measures​

    Supervised Learning Documentation. Official scikit-learn documentation about regression. Available at: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning​

    Different ways to compensate for missing values. Available at: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779​
    
    List of countries and dependencies by population. Available at: https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population
    
    Coronavirus Map: Tracking the Global Outbreak. Available at: https://www.nytimes.com/interactive/2020/world/coronavirus-maps.html

    COVID-19 Community Mobility Report. Available at: https://www.google.com/covid19/mobility/

    MEASURING OVERALL HEALTH SYSTEM PERFORMANCE. Available at: https://www.who.int/healthinfo/paper30.pdf?ua=1

    COVID-19: Government Response Stringency Index. Available at: https://ourworldindata.org/grapher/covid-stringency-index

    List of countries by GDP (PPP). Available at: https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)

    List of countries by Human Development Index. Available at: https://en.wikipedia.org/wiki/List_of_countries_by_Human_Development_Index

    List of countries by age structure. Available at: https://en.wikipedia.org/wiki/List_of_countries_by_age_structure
