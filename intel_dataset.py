import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

"""
This data source was from: https://www.kaggle.com/ltcptgeneral/cpu-specifications/data
The source contains specifications of CPUs in Intel and AMD's lineups respectively.

For this example, the Intel line was used, which can be found from the intel.csv

Features used for this clustering example: Cores, Cache Size, Base Speed
Target label is the status of the CPU (ie. Launched, Discontinued, Announced)
"""

data = pd.read_csv("data_source/intel.csv")

#This was done to randomise the order of the data points every time the script is run.
#Since the data points are not time-dependent, randomising their order would not affect the data integrity.
data = data.sample(frac = 1).reset_index(drop = True)

"""
Cache Size data had appendages such as 'KB' and 'MB'.
Since it is uncommon for Cache Sizes to be less than 1MB, CPUs with such Cache Sizes were excluded.
"""
data['Cache Size'] = data['Cache Size'].str.rstrip(' MB')
data = data[~data['Cache Size'].str.contains('K')]
#After removing the Cache Sizes in 'KB', I converted the values to floats to be used as inputs.
data['Cache Size'] = data['Cache Size'].astype(float)


"""
Base Speed data had appendages such as 'MHz' and 'GHz'.
Since it is uncommon for Base Speeds to be less than 1GHz in modern settings, CPUs with such Base Speeds were excluded.

In addition, there were entries without a Base Speed. Such entries were removed from the inputs.
"""
data = data[data['Base Speed'].notna()]
data['Base Speed'] = data['Base Speed'].str.rstrip('GHz')
data = data[~data['Base Speed'].str.contains(' M')]
#After removing the Base Speeds in 'MHz', I converted the values to floats to be used as inputs.
data['Base Speed'] = data['Base Speed'].astype(float)


"""
Cores data were a range of positive integers.
As such, no further removal / cleaning of data was required.
"""


"""
Status had the following values: 'Announced', 'Discontinued' and 'Launched'.
Since these were presesnted as strings, they had to be converted into numerical values to be fed as labels to the corresponding feature sets.
"""
data['Status'] = data['Status'].astype('category')
data['Status'] = data['Status'].cat.codes

#Category codes: 0 - Announced, 1 - Discontinued, 2 - Launched

"""
Rounding off the data points to 1490, as compared to a previously resulting number, which would be hard to split into training and testing sets later on.
Features were normalized based on the min. and max. of the training set to prevent data leakage.
"""
features = data[['Cores', 'Cache Size', 'Base Speed']][:1490]
labels = data['Status'][:1490]

train_length = len(features) * 0.8

#Splitting the data before normalizing
features_train = features.iloc[:int(train_length):]
features_test = features.iloc[int(train_length)::]

labels_train = labels.iloc[:int(train_length):]
labels_test = labels.iloc[int(train_length)::]

#Normalizing the data to a range of [0, 1]
cleaned_features_0_max = features_train['Cores'].max()
cleaned_features_0_min = features_train['Cores'].min()

cleaned_features_1_max = features_train['Cache Size'].max()
cleaned_features_1_min = features_train['Cache Size'].min()

cleaned_features_2_max = features_train['Base Speed'].max()
cleaned_features_2_min = features_train['Base Speed'].min()

features_train['Cores'] = (features_train['Cores'] - cleaned_features_0_min) / (cleaned_features_0_max - cleaned_features_0_min)
features_train['Cache Size'] = (features_train['Cache Size'] - cleaned_features_1_min) / (cleaned_features_1_max - cleaned_features_1_min)
features_train['Base Speed'] = (features_train['Base Speed'] - cleaned_features_2_min) / (cleaned_features_2_max - cleaned_features_2_min)


features_test['Cache Size'] = (features_test['Cache Size'] - cleaned_features_0_min) / (cleaned_features_0_max - cleaned_features_0_min)
features_test['Base Speed'] = (features_test['Base Speed'] - cleaned_features_1_min) / (cleaned_features_1_max - cleaned_features_1_min)
features_test['Cores'] = (features_test['Cores'] - cleaned_features_2_min) / (cleaned_features_2_max - cleaned_features_2_min)

X_train = features_train.values.tolist()
X_test = features_test.values.tolist()

y_train = labels_train.values.tolist()
y_test = labels_test.values.tolist()

