import matplotlib.pyplot as plt
import numpy as np

"""
This is my implementation of data preprocessing for use in my kNN implementation.

**THIS LIBRARY IS NOT USED IN THE FINAL CALCULATIONS**

Features include:
    1) Inputs are plain python lists or list of lists. 
    2) Returned values are converted to numpy arrays, so no further pre-processing needs to be done after using this library.
    3) Variable split index, default is set to 80-20.
    4) Scatter plot of features (2D) - compare 2 features at a time, which can be varied by the user.
    5) Scatter plots can be refined to display: all points (training + testing), only training or only testing.
"""

class TrainTestSplit:
    def __init__(self, features, labels):
        """
        Args:
            features ([array]): list of lists where each sublist represents a single data point
            labels ([array]): list that represents the labels for the corresponding data points

        NOTE: arrays should be a Python array, NOT a numpy array, as this class converts the arrays accordingly for you
        """
        
        if len(features) != len(labels):
            print("TrainTestSplit: Mismatch in the length of features and labels. Please check that they are of the same length.")

        self.features = features
        self.labels = labels
    
    def split(self, train_percentage = 0.8):
        split_index = (len(self.features) * train_percentage)

        if not split_index.is_integer():
            print("TrainTestSplit: Split percentage provided is not valid for the length of your dataset. Please try again")
            return
        else:
            confirmed_index = int(split_index)
            print("Training set ends at index {}... Splitting...".format(confirmed_index - 1))
            self.X_train = np.array(self.features[:confirmed_index])
            self.X_test = np.array(self.features[confirmed_index:])
            self.y_train = np.array(self.labels[:confirmed_index])
            self.y_test = np.array(self.labels[confirmed_index:])

            return(self.X_train, self.X_test, self.y_train, self.y_test)
    
    def plot_all(self, x_axis = 0, y_axis = 1):
        plot = plt.scatter(x = [x[x_axis] for x in self.features], y = [x[y_axis] for x in self.features], c = [y for y in self.labels])
        plt.title("Entire dataset")
        plt.legend(*plot.legend_elements(), loc = 4)
        plt.show()

    def plot_training(self, x_axis = 0, y_axis = 1):
        plot = plt.scatter(x = [x[x_axis] for x in self.X_train], y = [x[y_axis] for x in self.X_train], c = [y for y in self.y_train])
        plt.title("Training dataset")
        plt.legend(*plot.legend_elements(), loc = 4)
        plt.show()

    def plot_test(self, x_axis = 0, y_axis = 1):
        plot = plt.scatter(x = [x[x_axis] for x in self.X_test], y = [x[y_axis] for x in self.X_test], c = [y for y in self.y_test])
        plt.title("Test dataset")
        plt.legend(*plot.legend_elements(), loc = 4)
        plt.show()