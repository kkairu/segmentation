
# Simple Customer Segmentation Using Recency/Monetary Matrix

The RFM model:
- Recency: When was the last time they purchased?
- Frequency: How often and for how long have they purchased?
- Monetary Value/Sales: How much have they purchased?

Data Source - Global Superstore data by Tableau


```python
import matplotlib as plt
%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
```


```python
datafile = 'data\Global_Superstore.xls'

df = pd.read_excel(datafile)

# Filter data by consumer segment & country USA due to potential demographic bias
df = df[(df.Segment == 'Consumer') & (df.Country == 'United States')]
df.head()
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
      <th>Row ID</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Ship Date</th>
      <th>Ship Mode</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Segment</th>
      <th>City</th>
      <th>State</th>
      <th>...</th>
      <th>Product ID</th>
      <th>Category</th>
      <th>Sub-Category</th>
      <th>Product Name</th>
      <th>Sales</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Profit</th>
      <th>Shipping Cost</th>
      <th>Order Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32298</td>
      <td>CA-2012-124891</td>
      <td>2012-07-31</td>
      <td>2012-07-31</td>
      <td>Same Day</td>
      <td>RH-19495</td>
      <td>Rick Hansen</td>
      <td>Consumer</td>
      <td>New York City</td>
      <td>New York</td>
      <td>...</td>
      <td>TEC-AC-10003033</td>
      <td>Technology</td>
      <td>Accessories</td>
      <td>Plantronics CS510 - Over-the-Head monaural Wir...</td>
      <td>2309.650</td>
      <td>7</td>
      <td>0.0</td>
      <td>762.1845</td>
      <td>933.57</td>
      <td>Critical</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40936</td>
      <td>CA-2012-116638</td>
      <td>2012-01-28</td>
      <td>2012-01-31</td>
      <td>Second Class</td>
      <td>JH-15985</td>
      <td>Joseph Holt</td>
      <td>Consumer</td>
      <td>Concord</td>
      <td>North Carolina</td>
      <td>...</td>
      <td>FUR-TA-10000198</td>
      <td>Furniture</td>
      <td>Tables</td>
      <td>Chromcraft Bull-Nose Wood Oval Conference Tabl...</td>
      <td>4297.644</td>
      <td>13</td>
      <td>0.4</td>
      <td>-1862.3124</td>
      <td>865.74</td>
      <td>Critical</td>
    </tr>
    <tr>
      <th>21</th>
      <td>31784</td>
      <td>CA-2011-154627</td>
      <td>2011-10-29</td>
      <td>2011-10-31</td>
      <td>First Class</td>
      <td>SA-20830</td>
      <td>Sue Ann Reed</td>
      <td>Consumer</td>
      <td>Chicago</td>
      <td>Illinois</td>
      <td>...</td>
      <td>TEC-PH-10001363</td>
      <td>Technology</td>
      <td>Phones</td>
      <td>Apple iPhone 5S</td>
      <td>2735.952</td>
      <td>6</td>
      <td>0.2</td>
      <td>341.9940</td>
      <td>752.51</td>
      <td>High</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32735</td>
      <td>CA-2012-139731</td>
      <td>2012-10-15</td>
      <td>2012-10-15</td>
      <td>Same Day</td>
      <td>JE-15745</td>
      <td>Joel Eaton</td>
      <td>Consumer</td>
      <td>Amarillo</td>
      <td>Texas</td>
      <td>...</td>
      <td>FUR-CH-10002024</td>
      <td>Furniture</td>
      <td>Chairs</td>
      <td>HON 5400 Series Task Chairs for Big and Tall</td>
      <td>2453.430</td>
      <td>5</td>
      <td>0.3</td>
      <td>-350.4900</td>
      <td>690.42</td>
      <td>High</td>
    </tr>
    <tr>
      <th>34</th>
      <td>32543</td>
      <td>CA-2011-168494</td>
      <td>2011-12-12</td>
      <td>2011-12-14</td>
      <td>Second Class</td>
      <td>NP-18700</td>
      <td>Nora Preis</td>
      <td>Consumer</td>
      <td>Fresno</td>
      <td>California</td>
      <td>...</td>
      <td>FUR-TA-10003473</td>
      <td>Furniture</td>
      <td>Tables</td>
      <td>Bretford Rectangular Conference Table Tops</td>
      <td>3610.848</td>
      <td>12</td>
      <td>0.2</td>
      <td>135.4068</td>
      <td>683.12</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# Create RFM Features
df_RFM = df.groupby('Customer ID').agg({'Order Date': lambda y: (df['Order Date'].max().date() - y.max().date()).days,
                                        'Order ID': lambda y: len(y.unique()),  
                                        'Sales': lambda y: round(y.sum(),2)})
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
df_RFM = df_RFM.sort_values('Monetary', ascending=False)
df_RFM.head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RB-19360</th>
      <td>96</td>
      <td>6</td>
      <td>15117.34</td>
    </tr>
    <tr>
      <th>AB-10105</th>
      <td>41</td>
      <td>10</td>
      <td>14473.57</td>
    </tr>
    <tr>
      <th>KL-16645</th>
      <td>47</td>
      <td>12</td>
      <td>14175.23</td>
    </tr>
    <tr>
      <th>SC-20095</th>
      <td>349</td>
      <td>9</td>
      <td>14142.33</td>
    </tr>
    <tr>
      <th>HL-15040</th>
      <td>43</td>
      <td>6</td>
      <td>12873.30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Automate segmentation using 80% quantile for Recency and Monetary

quantiles = df_RFM.quantile(q=[0.8]) 
print(quantiles)

df_RFM['R']=np.where(df_RFM['Recency']<=int(quantiles.Recency.values), 2, 1)
df_RFM['F']=np.where(df_RFM['Frequency']>=int(quantiles.Frequency.values), 2, 1)
df_RFM['M']=np.where(df_RFM['Monetary']>=int(quantiles.Monetary.values), 2, 1)
df_RFM.head()
```

         Recency  Frequency  Monetary
    0.8    222.2        8.0   4070.17
    




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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RB-19360</th>
      <td>96</td>
      <td>6</td>
      <td>15117.34</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>AB-10105</th>
      <td>41</td>
      <td>10</td>
      <td>14473.57</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>KL-16645</th>
      <td>47</td>
      <td>12</td>
      <td>14175.23</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>SC-20095</th>
      <td>349</td>
      <td>9</td>
      <td>14142.33</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>HL-15040</th>
      <td>43</td>
      <td>6</td>
      <td>12873.30</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate RFM scores and sort customers

# To do the 2 x 2 matrix we will only use Recency & Monetary
df_RFM['RMScore'] = df_RFM.M.map(str)+df_RFM.R.map(str)
df_RFM = df_RFM.reset_index()
df_RFM_SUM = df_RFM.groupby('RMScore').agg({'Customer ID': lambda y: len(y.unique()),
                                        'Frequency': lambda y: round(y.mean(),0),
                                        'Recency': lambda y: round(y.mean(),0),
                                        'R': lambda y: round(y.mean(),0),
                                        'M': lambda y: round(y.mean(),0),
                                        'Monetary': lambda y: round(y.mean(),0)})

df_RFM_SUM = df_RFM_SUM.sort_values('RMScore', ascending=False)
df_RFM_SUM.head()
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
      <th>Customer ID</th>
      <th>Frequency</th>
      <th>Recency</th>
      <th>R</th>
      <th>M</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>RMScore</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>73</td>
      <td>8</td>
      <td>59</td>
      <td>2</td>
      <td>2</td>
      <td>6711.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9</td>
      <td>7</td>
      <td>425</td>
      <td>1</td>
      <td>2</td>
      <td>8564.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>254</td>
      <td>6</td>
      <td>70</td>
      <td>2</td>
      <td>1</td>
      <td>1902.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>73</td>
      <td>4</td>
      <td>440</td>
      <td>1</td>
      <td>1</td>
      <td>1526.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the Value Matrix and explore

# 1) Average Monetary Matrix
df_RFM_M = df_RFM_SUM.pivot(index='M', columns='R', values='Monetary')
df_RFM_M= df_RFM_M.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
print(df_RFM_M)

```

    R       1       2
    M                
    2  8564.0  6711.0
    1  1526.0  1902.0
    


```python
# 2) Number of Customer Matrix
df_RFM_C = df_RFM_SUM.pivot(index='M', columns='R', values='Customer ID')
df_RFM_C= df_RFM_C.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
print(df_RFM_C)
```

    R   1    2
    M         
    2   9   73
    1  73  254
    


```python
# 3) Recency Matrix
df_RFM_R = df_RFM_SUM.pivot(index='M', columns='R', values='Recency')
df_RFM_R= df_RFM_R.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
print(df_RFM_R)
```

    R    1   2
    M         
    2  425  59
    1  440  70
    

![2x2matrix.png](attachment:2x2matrix.png)

# Takeaways 
- There are few customers in the “Disengaged” bucket and they have an average revenue higher than the “Star” bucket. Action is to coantact the customers and activate them. Engage
- The average last order from the “Light” bucket is very old (> 1 yr vs. 60-70 days for ‘engaged’ customers). Launch a simple reactivation campaign
