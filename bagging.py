"""
Each label has approx 50 images
"""
# TODO: Neural network is overfitting or not working

import numpy as np
import pandas as pd
from itertools import count

import global_variables
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from keras.layers import Input, Dense, Conv1D, Flatten, MaxPool1D, TimeDistributed, CuDNNLSTM, GlobalMaxPool2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping,ModelCheckpoint

import xgboost as xgb
from xgboost import XGBRegressor

from utilities import acc_prediction_valid, bag_by_average, bag_by_geomean, top1_loss


np.random.seed(0)
ROOT_DIRECTORY = '/Users/macuni/Desktop/FurnitureClassification/predictions/'
NUMBER_OF_CLASSES = 128

MODELS = ['models/Inceptionv3/Inception_LSTM/',
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

CLASSIFIERS = ['xgb', 'lr', 'nn_keras'] # 'nn' #, 'lgb'] #nn, catboost


class CSVPredictions():
    '''
    Reads the csvfiles and preprocesses the labels and predictions
    '''
    def __init__(self, models):
        self.csv_filenames = [ROOT_DIRECTORY + m + 'prediction_valid.csv' for m in models]
        self.dfs = [pd.read_csv(csv_file, index_col=0) for csv_file in self.csv_filenames]

        assert self.dfs[0].shape[1] == NUMBER_OF_CLASSES, 'Number of classes does not match.'

        # self.model_predictions has shape (len(models), self.number_of_samples, NUMBER_OF_CLASSES)
        # self.model_predictions has predictions according to the labels used in training, not according to the filenames
        self.model_predictions = np.array([df.values for df in self.dfs])
        _, self.number_of_samples, _ = self.model_predictions.shape

        # self.dfs[0].index has the filenames as entries
        self.labels = np.array([self._get_label(label.split('/')[0]) for label in self.dfs[0].index])

    def train_valid_split(self, ratio):
        split = int(ratio * self.number_of_samples)

        indices = np.arange(self.number_of_samples)
        np.random.shuffle(indices)
        # we need to transpose, split and untranspose TODO transpose consistently.
        predictions = np.copy(self.model_predictions)
        labels = np.copy(self.labels)

        #shuffle data, since predictions are ordered
        predictions = np.transpose(predictions, (1,0,2))
        predictions = predictions[indices]
        labels = labels[indices]

        train_pred, valid_pred = predictions[:split], predictions[split:]
        train_pred, valid_pred = np.transpose(train_pred, (1,0,2)), np.transpose(valid_pred, (1,0,2))

        return train_pred, valid_pred, labels[:split], labels[split:]

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

    def print_model_accuracy(self):
        for k, df in enumerate(self.dfs):
            print('Error for model {}:'.format(self.csv_filenames[k]))
            print(acc_prediction_valid(df))


class Bagging():
    '''
    Base class for bagging models
    '''
    def __init__(self, predictions, labels):
        # self.model_predictions has shape (len(models), self.number_of_samples, NUMBER_OF_CLASSES)
        self.model_predictions = predictions
        self.number_of_models, self.number_of_samples, _ = self.model_predictions.shape
        self.labels = labels
        # sparse labels has shape (self.number_of_samples, NUMBER_OF_CLASSES)
        self.sparse_labels = Bagging._convert_labels_to_sparse(self.labels)

    @staticmethod
    def _convert_labels_to_sparse(labels):
        '''
        Converts the labels to an array,
        :return:
        '''
        return np.array([[1 if i == label else 0 for i in range(NUMBER_OF_CLASSES)] for label in labels])

    def calculate_accuracy(self, predictions):
        """
        :param predictions: numpy array of shape (self.number_of_samples, NUMBER_OF_CLASSES)
        :return:
        """
        assert predictions.shape == (self.number_of_samples, NUMBER_OF_CLASSES), 'shape of predictions does not match.'

        correct_predictions = [np.argmax(prediction) == label for prediction, label in zip(predictions, self.labels)]
        return 1 - sum(correct_predictions) / self.number_of_samples

    def geometric_mean(self):
        return self.calculate_accuracy(bag_by_geomean(self.model_predictions))

    def average_mean(self):
        return self.calculate_accuracy(bag_by_average(self.model_predictions))

    def refined_average_mean(self):
        '''
        Calculate the mean according to weights.
        :return:
        '''

        """
        Error for model Inceptionv3/Inception_LSTM/prediction_valid.csv:
        #0.19559359644951657
        Error for model InceptionResNetV2/LSTM/prediction_valid.csv:
        0.1917895070534158
        Error for model DenseNet121/LSTM/prediction_valid.csv:
        0.20431130131558095
        Error for model Inceptionv3/Inception_Dense/prediction_valid.csv:
        0.21413853225550805
        Error for model Xception/GlobalPooling/prediction_valid.csv:
        0.1867173878586147
        Error for model Xception/Gru_256/prediction_valid.csv:
        0.18560786178475197
        Error for model Xception/Gru_512/prediction_valid.csv:
        0.18116975748930098
        Error for model Xception/LSTM2/prediction_valid.csv:
        0.1922650182279284
        Error for model DenseNet201/Last_layer/prediction_valid.csv:
        0.1623078142336345
        Error for model DenseNet201/Last_layer_with_dropout/prediction_valid.csv:
        0.157711206213346
        """

        weights = [1., 1., 0., 0., 1., 1., 1., 1., 2., 2.]
        weights /= np.sum(weights)

        bagged_predicts = np.zeros((self.number_of_samples, NUMBER_OF_CLASSES))
        for i, predict in enumerate(self.model_predictions):
            bagged_predicts += weights[i] * predict

        return self.calculate_accuracy(bagged_predicts)

    def correlations(self):
        return np.round(np.corrcoef(np.array([x.flatten() for x in self.model_predictions])), 2)


TEST_LABELS = [1,2,1,4,5,1,2,3,4,5]
TEST_LABELS_AS_SPASRE = Bagging._convert_labels_to_sparse(TEST_LABELS)
FIRST_HOT = np.array([1,0,1,0,0,0,0,0,0,0])
TWO_HOT =   np.array([0,1,0,0,0,0,1,0,0,0])

assert TEST_LABELS_AS_SPASRE[:, 0].all() == FIRST_HOT.all()
assert TEST_LABELS_AS_SPASRE[:, 1].all() == TWO_HOT.all()


class Regressions(Bagging):

    def __init__(self, predictions, labels):
        super().__init__(predictions, labels)

    def train_logistic_regression(self):
        return self._train_classifier(LogisticRegression)

    def predict_logistic_regression(self, classifiers):
        # generate for each sample a probability distribution.
        predict_logistic = np.zeros((self.number_of_samples, NUMBER_OF_CLASSES))
        for i in range(NUMBER_OF_CLASSES):
            x_train = np.transpose(self.model_predictions[:, :, i])
            # save only the probability of class being 1
            predict_logistic[:, i] = classifiers[i].predict_proba(x_train)[:, 1]

        predictions = normalize(predict_logistic, norm='l1', axis=1, copy=True, return_norm=False)

        return predictions

    def _train_classifier(self, classifier, **kwargs):
        classifiers = []

        for i in range(NUMBER_OF_CLASSES):
            print('fitting on class {}'.format(i))
            clf = classifier(**kwargs)
            # transpose for shape (self.number_of_samples, len(models))
            x_train = np.transpose(self.model_predictions[:, :, i])
            y_train = self.sparse_labels[:, i]
            clf.fit(x_train, y_train)
            classifiers.append(clf)
        return classifiers

    def train_xgb(self):
        return self._train_classifier(XGBRegressor, objective='reg:logistic', max_depth=2, n_estimators=100,\
                                      learning_rate=0.1, subsample=0.8, min_child_weight=3)

    def predict_xgb(self, classifiers):
        predict_xgb = np.zeros((self.number_of_samples, NUMBER_OF_CLASSES))
        for i in range(NUMBER_OF_CLASSES):
            x_train = np.transpose(self.model_predictions[:, :, i])
            predict_xgb[:, i] = classifiers[i].predict(x_train)

        predictions = normalize(predict_xgb, norm='l1', axis=1, copy=True, return_norm=False)

        return predictions


class NeuralNetwork(Bagging):

    def __init__(self, predictions, labels):
        super().__init__(predictions, labels)

    def train_neural_network(self):
        clf_nn = MLPClassifier(solver='adam', batch_size=256, hidden_layer_sizes=(256,), max_iter=200, verbose=True,
                               tol=0.000001, )
        clf_nn.loss = 'log_loss'
        # reshape input such that it is (self.number_of_samples, len(models), NUMBER_OF_CLASSES)
        # each training data has len(models) * self.number_of_samples = 10 * 128 dimensions
        x_train = np.reshape(np.copy(self.model_predictions), (self.number_of_samples, self.number_of_models * NUMBER_OF_CLASSES))
        sparse_labels = np.copy(self.sparse_labels)
        clf_nn.fit(x_train, sparse_labels)

        return clf_nn

    def predict_neural_network(self, clf_nn):
        x_valid = np.reshape(self.model_predictions, (self.number_of_samples, self.number_of_models * NUMBER_OF_CLASSES))
        nn_predictions = clf_nn.predict_proba(x_valid)
        assert nn_predictions.shape == (self.number_of_samples, NUMBER_OF_CLASSES)
        return nn_predictions

    def train_keras_neural_network(self):
        batchsize = 256
        hidden_layer_size = 256
        dropout = 0.4
        learning_rate = 0.01
        epochs = 50
        train_valid_split = 0.9

        inp = Input(shape=(self.number_of_models * NUMBER_OF_CLASSES,))
        hidden = Dense(hidden_layer_size, activation='sigmoid')(inp)
        hidden_dropout = layers.Dropout(dropout)(hidden)
        out = Dense(NUMBER_OF_CLASSES, activation='sigmoid')(hidden_dropout)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=[top1_loss])
        model.summary()

        check_point = ModelCheckpoint('models/Inceptionv3/Inception_LSTM/best.hdf5', monitor="val_loss", mode="min",
                                      save_best_only=True, verbose=1)
        early_stop = EarlyStopping(patience=7)
        model.fit_generator(self._get_batch(batchsize, 'train', train_valid_split),
                            validation_data=self._get_batch(batchsize, 'valid', train_valid_split),
                            callbacks=[check_point],
                            validation_steps=int(self.number_of_samples * (1 - train_valid_split) / batchsize),
                            steps_per_epoch=int(self.number_of_samples * train_valid_split / batchsize),
                            epochs=epochs,
                            verbose=0)

        X = np.reshape(self.model_predictions, (self.number_of_samples, self.number_of_models * NUMBER_OF_CLASSES))
        return model

    def predict_keras_neural_network(self, nn_model):
        X = np.reshape(self.model_predictions, (self.number_of_samples, self.number_of_models * NUMBER_OF_CLASSES))
        return nn_model.predict(X, batch_size=None, verbose=0, steps=None)

    def _get_batch(self, batchsize, mode, split):
        assert mode in ['train', 'valid']
        split_size = int(split * self.number_of_samples)

        # copy self.model_predictions and self.sparse_labels due to in-place shuffling
        complete_x = np.reshape(np.copy(self.model_predictions), (self.number_of_samples, self.number_of_models * NUMBER_OF_CLASSES))
        X = complete_x[:split_size] if mode == 'train' else complete_x[split_size:]
        labels = np.copy(self.sparse_labels)[:split_size] if mode == 'train' else np.copy(self.sparse_labels)[split_size:]

        assert labels.shape[0] >= batchsize, 'Batchsize chosen too large'

        for epoch in count():
            indices = np.arange(labels.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            labels = labels[indices]

            for i in range(X.shape[0] // batchsize):
                start = i * batchsize
                end = (i + 1) * batchsize
                yield X[start:end], labels[start:end]


def average_errors():
    bagging= Bagging(csv_models.model_predictions, csv_models.labels)
    print('Correlations:')
    print(bagging.correlations())
    print('Error on refined mean: {}'.format(bagging.refined_average_mean()))
    print('Error on average mean: {}'.format(bagging.average_mean()))
    print('Error on geometric mean: {}'.format(bagging.geometric_mean()))


def logistic_regression(train_pred, valid_pred, train_labels, valid_labels):
    print('Logistic regression:')
    train_classifier = Regressions(train_pred, train_labels)
    validator = Regressions(valid_pred, valid_labels)

    logistic_classifiers = train_classifier.train_logistic_regression()
    logistic_predictions = validator.predict_logistic_regression(logistic_classifiers)
    error = validator.calculate_accuracy(logistic_predictions)
    print('Error on logistic regression: {}'.format(error))
    return error, logistic_predictions


def xgb(train_pred, valid_pred, train_labels, valid_labels):
    print('Xgb')
    train_classifier = Regressions(train_pred, train_labels)
    validator = Regressions(valid_pred, valid_labels)

    xgb_classifiers = train_classifier.train_xgb()
    xgb_predictions = validator.predict_xgb(xgb_classifiers)
    error = validator.calculate_accuracy(xgb_predictions)
    print('Error on xgb: {}'.format(error))
    return error, xgb_predictions


def neural_network(train_pred, valid_pred, train_labels, valid_labels):
    print('Neural Network')
    train_classifier = NeuralNetwork(train_pred, train_labels)
    validator = NeuralNetwork(valid_pred, valid_labels)

    nn_model = train_classifier.train_neural_network()
    nn_predictions = validator.predict_neural_network(nn_model)
    error = validator.calculate_accuracy(nn_predictions)
    print('Error on neural network: {}'.format(error))
    return error, nn_predictions


def neural_netwokr_keras(train_pred, valid_pred, train_labels, valid_labels):
    print('Neural Network with Keras')
    train_classifier = NeuralNetwork(train_pred, train_labels)
    validator = NeuralNetwork(valid_pred, valid_labels)

    print('Neural Network with Keras')
    nn_model_keras = train_classifier.train_keras_neural_network()
    nn_predictions_keras = validator.predict_keras_neural_network(nn_model_keras)
    error = validator.calculate_accuracy(nn_predictions_keras)
    print('Error on keras neural network: {}'.format(error))
    return error, nn_predictions_keras


if __name__ == '__main__':
    csv_models = CSVPredictions(MODELS)
    predictions, labels = csv_models.model_predictions, csv_models.labels
    train_pred, valid_pred, train_labels, valid_labels = csv_models.train_valid_split(0.9)

    errors = {}

    if 'lr' in CLASSIFIERS:
        lr_error, lr_predictions = logistic_regression(train_pred, valid_pred, train_labels, valid_labels)
        errors['logistic_regression'] = lr_error

    if 'nn' in CLASSIFIERS:
        nn_error, nn_predictions = neural_network(train_pred, valid_pred, train_labels, valid_labels)
        errors['neural network'] = nn_error

    if 'nn_keras' in CLASSIFIERS:
        nn_keras_error, nn_keras_predictions = neural_netwokr_keras(train_pred, valid_pred, train_labels, valid_labels)
        errors['neural network keras'] = nn_keras_error

    if 'xgb' in CLASSIFIERS:
        xgb_error, xgb_predictions = xgb(train_pred, valid_pred, train_labels, valid_labels)
        errors['xgb'] = xgb_error

    csv_models.print_model_accuracy()
    average_errors()
    print(errors)