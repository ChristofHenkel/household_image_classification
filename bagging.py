"""
Each label has approx 50 images
"""

import pandas as pd
import global_variables
from utilities import acc_prediction_valid, bag_by_average, bag_by_geomean
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np


ROOT_DIRECTORY = '/Users/macuni/Desktop/FurnitureClassification/predictions/'
NUMBER_OF_CLASSES = 128

models = ['models/Inceptionv3/Inception_LSTM/',
          'models/InceptionResNetV2/LSTM/',
          'models/DenseNet121/LSTM/',
          'models/Inceptionv3/Inception_Dense/',
          'models/Xception/GlobalPooling/',
          'models/Xception/Gru_256/',
          'models/Xception/Gru_512/',
          'models/Xception/LSTM2/',
          #'models/MobileNet/BiLSTM_256/',
          #'models/MobileNet/SeperableConv/',
          'models/DenseNet201/Last_layer/',
          'models/DenseNet201/Last_layer_with_dropout/'
          ]


methods = ['nn', 'lr']

class Bagging():

    def __init__(self, models):
        self.csv_filenames = [ROOT_DIRECTORY + m + 'prediction_valid.csv' for m in models]
        self.dfs = [pd.read_csv(csv_file, index_col=0) for csv_file in self.csv_filenames]
        self.number_of_samples = self.dfs[0].shape[0]

        assert self.dfs[0].shape[1] == NUMBER_OF_CLASSES, 'Number of classes does not match.'

        # self.model_predictions has shape (len(models), self.number_of_samples, NUMBER_OF_CLASSES)
        # self.model_predictions has predictions according to the labels used in training, not according to the filenames
        self.model_predictions = np.array([df.values for df in self.dfs])
        # self.dfs[0].index has the filenames as entries
        self.labels = np.array([self._get_label(label.split('/')[0]) for label in self.dfs[0].index])
        # sparse labels has shape (self.number_of_samples, NUMBER_OF_CLASSES)
        self.sparse_labels = Bagging._convert_labels_to_sparse(self.labels)

    @staticmethod
    def _convert_labels_to_sparse(labels):
        '''
        Converts the labels to an array,
        :return:
        '''
        return np.array([[1 if i == label else 0 for i in range(NUMBER_OF_CLASSES)] for label in labels])

    def _get_label(self, label_from_file):
        '''
        helper function to convert labels from filename to labels used in training.
        :param label_from_file: label from file
        :return: label used in training
        '''
        label2labelid = \
            {'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '103': 5, '104': 6, '105': 7, '106': 8, '107': 9, '108': 10,
             '109': 11, '11': 12, '110': 13, '111': 14, '112': 15, '113': 16, '114': 17, '115': 18, '116': 19,
             '117': 20,
             '118': 21, '119': 22, '12': 23, '120': 24, '121': 25, '122': 26, '123': 27, '124': 28, '125': 29,
             '126': 30,
             '127': 31, '128': 32, '13': 33, '14': 34, '15': 35, '16': 36, '17': 37, '18': 38, '19': 39, '2': 40,
             '20': 41,
             '21': 42, '22': 43, '23': 44, '24': 45, '25': 46, '26': 47, '27': 48, '28': 49, '29': 50, '3': 51,
             '30': 52,
             '31': 53, '32': 54, '33': 55, '34': 56, '35': 57, '36': 58, '37': 59, '38': 60, '39': 61, '4': 62,
             '40': 63,
             '41': 64, '42': 65, '43': 66, '44': 67, '45': 68, '46': 69, '47': 70, '48': 71, '49': 72, '5': 73,
             '50': 74,
             '51': 75, '52': 76, '53': 77, '54': 78, '55': 79, '56': 80, '57': 81, '58': 82, '59': 83, '6': 84,
             '60': 85,
             '61': 86, '62': 87, '63': 88, '64': 89, '65': 90, '66': 91, '67': 92, '68': 93, '69': 94, '7': 95,
             '70': 96,
             '71': 97, '72': 98, '73': 99, '74': 100, '75': 101, '76': 102, '77': 103, '78': 104, '79': 105, '8': 106,
             '80': 107, '81': 108, '82': 109, '83': 110, '84': 111, '85': 112, '86': 113, '87': 114, '88': 115,
             '89': 116,
             '9': 117, '90': 118, '91': 119, '92': 120, '93': 121, '94': 122, '95': 123, '96': 124, '97': 125,
             '98': 126,
             '99': 127}
        return label2labelid[label_from_file]

    def calculate_accuracy(self, predictions):
        """
        :param predictions: numpy array of shape (self.number_of_samples, NUMBER_OF_CLASSES)
        :return:
        """
        assert predictions.shape == (self.number_of_samples, NUMBER_OF_CLASSES), 'shape of predictions does not match.'

        correct_predictions = [np.argmax(prediction) == label for prediction, label in zip(predictions, self.labels)]
        return 1 - sum(correct_predictions) / self.number_of_samples


    def _deprecated_predict_accuracy(self, predictions):
        '''

        :param predictions: List containing numpy array of shape (Validation_images , Classes) (equal to (6309, 128))
        predictions[i] gives the probability distribution for a specific image
        :return:
        '''
        df_new = self.dfs[0].copy()
        df_new[df_new.columns] = predictions
        return acc_prediction_valid(df_new)

    def logistic_regression(self):
        self.lr_clfs = []

        for i in range(NUMBER_OF_CLASSES):
            print('fitting logistic regression on class {}'.format(i))
            clf_lr = LogisticRegression()
            # transpose for shape (self.number_of_samples, len(models))
            x_train = np.transpose(self.model_predictions[:, :, i])
            y_train = self.sparse_labels[:, i]
            clf_lr.fit(x_train, y_train)
            self.lr_clfs.append(clf_lr)

        # generate for each sample a probability distribution.
        predict_logistic = np.zeros((self.number_of_samples, NUMBER_OF_CLASSES))
        for i in range(NUMBER_OF_CLASSES):
            x_train = np.transpose(self.model_predictions[:, :, i])
            # save only the probability of class being 1
            predict_logistic[:, i] = self.lr_clfs[i].predict_proba(x_train)[:, 1]

        predictions = normalize(predict_logistic, norm='l1', axis=1, copy=True, return_norm=False)

        return predictions

    def neural_network(self):
        clf_nn = MLPClassifier(solver='adam', batch_size=256, hidden_layer_sizes=(512,), max_iter=2000, verbose=True,
                               tol=0.000001, )
        clf_nn.loss = 'log_loss'
        # reshape input such that it is (self.number_of_samples, len(models), NUMBER_OF_CLASSES)
        # each training data has len(models) * self.number_of_samples = 10 * 128 dimensions
        x_train = np.reshape(self.model_predictions, (self.number_of_samples, len(models) * NUMBER_OF_CLASSES))
        clf_nn.fit(x_train, self.sparse_labels)

        nn_predictions = clf_nn.predict_proba(x_train)
        assert nn_predictions.shape == (self.number_of_samples, NUMBER_OF_CLASSES)
        return nn_predictions

    def geometric_mean(self):
        return self.calculate_accuracy(bag_by_geomean(self.model_predictions))

    def average_mean(self):
        return self.calculate_accuracy(bag_by_average(self.model_predictions))

    def refined_average_mean(self):
        '''
        Calculate the mean according to the individual accuracy
        :return:
        '''
        accuracies = np.array([1 - acc_prediction_valid(df) for df in self.dfs])
        accuracies /= np.sum(accuracies)

        bagged_predicts = np.zeros((self.number_of_samples, NUMBER_OF_CLASSES))
        for i, predict in enumerate(self.model_predictions):
            bagged_predicts += accuracies[i] * predict

        return self.calculate_accuracy(bagged_predicts)

    def print_model_accuracy(self):
        for k, df in enumerate(self.dfs):
            print('Error for model {}:'.format(self.csv_filenames[k]))
            print(acc_prediction_valid(df))

    def correlations(self):
        return np.round(np.corrcoef(np.array([x.flatten() for x in self.model_predictions])), 2)


TEST_LABELS = [1,2,1,4,5,1,2,3,4,5]
TEST_LABELS_AS_SPASRE = Bagging._convert_labels_to_sparse(TEST_LABELS)
FIRST_HOT = np.array([1,0,1,0,0,0,0,0,0,0])
TWO_HOT =   np.array([0,1,0,0,0,0,1,0,0,0])

assert TEST_LABELS_AS_SPASRE[:, 0].all() == FIRST_HOT.all()
assert TEST_LABELS_AS_SPASRE[:, 1].all() == TWO_HOT.all()


if __name__ == '__main__':
    bagging = Bagging(models)

    bagging.print_model_accuracy()

    if 'lr' in methods:
        logistic_prediction = bagging.logistic_regression()
        print('Error on logistic regression: {}'.format(bagging.calculate_accuracy(logistic_prediction)))

    if 'nn' in methods:
        print('Neural Network')
        nn_predictions = bagging.neural_network()
        print('Error on logistic regression: {}'.format(bagging.calculate_accuracy(nn_predictions)))

    print('Correlations:')
    print(bagging.correlations())
    print('Error on refined mean: {}'.format(bagging.refined_average_mean()))
    print('Error on average mean: {}'.format(bagging.average_mean()))
    print('Error on geometric mean: {}'.format(bagging.geometric_mean()))
