from collections import Counter
import numpy as np

"""
This is my implementation of k-Nearest Neighbours.

Features include:
    1) Variable value of k
    2) Changing the distance measurement used (Euclidean vs Manhattan) - uses majority voting
    3) Showing the test accuracy immediately after training is complete

Some things to note before usage:
    1) Numpy functions are used (sum, sqrt) as they process arrays much faster than implementing nested for loops
    2) Counter function is used instead of list.count() as it is simpler to implement, and is faster as the number of points grows
"""

class kNearestNeighbours:
    def __init__(self, k, distance_method = 'euclidean'):
        """representation of the k-Nearest Neighbours implementation

        Args:
            k ([int]): represents the k value to be used when computing the majority vote
            distance_method (str, optional): Choose either 'euclidean' or 'manhattan' distance computations. Defaults to 'euclidean'.
        """
        self.k = k
        self.distance_method = distance_method

    def fit(self, X_train, y_train, X_test, y_test):
        """PLEASE NOTE: all inputs should be numpy arrays

        Args:
            X_train ([array]): list of lists of features for training the kNN model
            y_train ([array]): list of labels for training the kNN model
            X_test ([array]): list of lists of features for testing the accuracy of the model
            y_test ([array]): list of labels for testing the accuracu of the model
        """
        self.X_train  = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        test_labels = self.predicted_labels(self.X_test)
        test_accuracy = self.accuracy(y_true = test_labels, y_pred = self.y_test)

        print("Test Accuracy: {}".format(test_accuracy))
    
    def distance_euclidean(self, x1, x2):
        """
        Args:
            x1 ([array]): feature array of test / input data
            x2 ([array]): feature array of training data

        Returns:
            [float]: euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def distance_manhattan(self, x1, x2):
        """
        Args:
            x1 ([array]): feature array of test / input data
            x2 ([array]): feature array of training data

        Returns:
            [float]: manhattan distance between x1 and x2
        """
        return np.sum(abs(x1 - x2))

    def prediction(self, x):
        """predicts the labels for the given input features

        Args:
            x ([numpy array]): single numpy array that is computed against the whole training set

        Returns:
            [array]: predicted label
        """
        if self.distance_method == 'euclidean':
            distance_between_train = [self.distance_euclidean(x, x_training) for x_training in self.X_train]
        elif self.distance_method == 'manhattan':
            distance_between_train = [self.distance_manhattan(x, x_training) for x_training in self.X_train]

        indices = np.argsort(distance_between_train)[:self.k]

        nearest_labels = [self.y_train[i] for i in indices]

        majority_vote = Counter(nearest_labels).most_common(1)

        return majority_vote[0][0]

    def predicted_labels(self, x):
        """
        Args:
            x ([numpy array]): numpy array of feature sets to be used for predictions

        Returns:
            [numpy array]: array of predicted labels
        """
        predictions = [self.prediction(x_indiv) for x_indiv in x]

        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        """
        Args:
            y_true ([numpy array]): array of ground truth labels
            y_pred ([numpy array]): array of predicted labels

        Returns:
            [float]: accuracy based on true values against predicted values
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        
        return accuracy
