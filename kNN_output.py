import numpy as np

from preprocessing import TrainTestSplit
from kNN import kNearestNeighbours
import intel_dataset


#splitting data
data_split = TrainTestSplit(intel_dataset.cleaned_features, intel_dataset.cleaned_labels)

X_train, X_test, y_train, y_test = data_split.split()

#initializing kNN classifier
knn = kNearestNeighbours(k = 3)

knn.fit(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

#input features are as follows: ['Cores', 'Cache Size', 'Base Speed']

raw_input = [
    [28, 30, 2.80],
    [24, 25, 2.94]
    ]

normalized_input = []

for features in raw_input:
    normalized_feature = [
        (features[0] - intel_dataset.cleaned_features_0_min) / (intel_dataset.cleaned_features_0_max - intel_dataset.cleaned_features_0_min),
        (features[1] - intel_dataset.cleaned_features_1_min) / (intel_dataset.cleaned_features_1_max - intel_dataset.cleaned_features_1_min),
        (features[2] - intel_dataset.cleaned_features_2_min) / (intel_dataset.cleaned_features_2_max - intel_dataset.cleaned_features_2_min),
    ]

    normalized_input.append(normalized_feature)

predictions = knn.predicted_labels(x = np.array(normalized_input))

#returns a numpy array of predictions
print(predictions)

#placed at the end to ensure that graphs display after model has been trained
data_split.plot_all(x_axis = 1, y_axis = 2)
data_split.plot_all(x_axis = 0, y_axis = 2)
