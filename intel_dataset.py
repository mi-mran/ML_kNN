import pandas as pd

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
#Calculation of min. and max. of the feature is done here since these values would be affected once normalized.
cleaned_features_1_max = data['Cache Size'].max()
cleaned_features_1_min = data['Cache Size'].min()
#Normalizing the data to a range of [0, 1]
data['Cache Size'] = (data['Cache Size'] - data['Cache Size'].min()) / (data['Cache Size'].max() - data['Cache Size'].min())

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
#Calculation of min. and max. of the feature is done here since these values would be affected once normalized.
cleaned_features_2_max = data['Base Speed'].max()
cleaned_features_2_min = data['Base Speed'].min()
#Normalizing the data to a range of [0, 1]
data['Base Speed'] = (data['Base Speed'] - data['Base Speed'].min()) / (data['Base Speed'].max() - data['Base Speed'].min())


"""
Cores data were a range of positive integers.
As such, no further removal / cleaning of data was required.
"""
#Calculation of min. and max. of the feature is done here since these values would be affected once normalized.
cleaned_features_0_max = data['Cores'].max()
cleaned_features_0_min = data['Cores'].min()
#Normalizing the data to a range of [0, 1]
data['Cores'] = (data['Cores'] - data['Cores'].min()) / (data['Cores'].max() - data['Cores'].min())

"""
Status had the following values: 'Announced', 'Discontinued' and 'Launched'.
Since these were presesnted as strings, they had to be converted into numerical values to be fed as labels to the corresponding feature sets.
"""
data['Status'] = data['Status'].astype('category')
data['Status'] = data['Status'].cat.codes

#Category codes: 0 - Announced, 1 - Discontinued, 2 - Launched

"""
Rounding off the data points to 1490, as compared to a previously resulting number, which would be hard to split into training and testing sets later on.
Features and labels were converted to lists, cleaned_features is a list of lists and cleaned_labels is a single list.
"""
cleaned_features = data[['Cores', 'Cache Size', 'Base Speed']][:1490].values.tolist()

cleaned_labels = data['Status'][:1490].values.tolist()





