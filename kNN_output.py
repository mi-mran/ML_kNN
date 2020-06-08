from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import TrainTestSplit
from kNN import kNearestNeighbours
import intel_dataset

#initializing kNN classifier
knn = kNearestNeighbours(k = 3)

knn.fit(
    X_train = np.array(intel_dataset.X_train), 
    y_train = np.array(intel_dataset.y_train), 
    X_test = np.array(intel_dataset.X_test), 
    y_test = np.array(intel_dataset.y_test)
    )

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

fig = plt.figure()
ax = fig.gca(projection='3d')

scatter_plot = ax.scatter(
    xs = [x[0] for x in (intel_dataset.X_train + intel_dataset.X_test)], 
    ys = [x[1] for x in (intel_dataset.X_train + intel_dataset.X_test)],  
    zs = [x[2] for x in (intel_dataset.X_train + intel_dataset.X_test)], 
    c = [y for y in (intel_dataset.y_train + intel_dataset.y_test)],
    )
plt.title("Entire dataset")
ax.set_xlabel('Cores')
ax.set_ylabel('Cache Size')
ax.set_zlabel('Base Speed')

plt.show()