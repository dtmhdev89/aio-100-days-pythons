import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from dataset_man import dataset_manager

def compute_error(y, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation


    Note that all arrays should be the same length
    '''

    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    espilon = 0.01
    return np.log((1 - error + espilon) / (error+espilon))

def update_weights_formular1(w_i, alpha, y, y_pred):
    # for case labels {-1, 1}. y and y_predict either receives value of -1 or 1 to increase if wrong prediction, or decrease if right precition

    result = w_i * np.exp(-alpha * y * y_pred)
    w_norm = result / np.sum(result)
    return w_norm

def update_weights_formular2(w_i, alpha, y, y_pred):
    # for case labels {0, 1}.

    result = w_i * np.exp(alpha * (
        np.not_equal(y, y_pred)).astype(int))
    w_norm = result / np.sum(result)
    return w_norm

# Define AdaBoost class
class AIVNAdaBoost:

    def __init__(self):
        # self.w_i = None
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, M = 100):
        '''
        Fit model. Arguments:
        X: independent variables
        y: target variable
        M: number of boosting rounds. Default is 100
        '''

        # Clear before calling
        self.alphas = []
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M):

            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                 w_i = update_weights_formular2(w_i, alpha_m, y, y_pred)
                # w_i = update_weights_formular1(w_i, alpha_m, y, y_pred)
            # print(w_i)

            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)
            # print(error_m)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)
            # print(alpha_m)

        assert len(self.G_M) == len(self.alphas)
    
    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables
        '''

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M))

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            #weak_preds.iloc[:,m] = y_pred_m
            weak_preds[weak_preds.columns[m]] = y_pred_m

        # Estimate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
    
    def error_rates(self, X, y):
        '''
        Get the error rates of each weak classifier. Arguments:
        X: independent variables
        y: target variables associated to X
        '''

        self.prediction_errors = [] # Clear before calling

        # Predict class label for each weak classifier
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X)
            error_m = compute_error(y = y, y_pred = y_pred_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)

def main():
    #Prepare dataset
    X, y = make_classification(n_samples= 1000, n_features = 20, random_state = 42)

    y = y * 2 - 1       # Original AdaBoost uses {1, -1} as class labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Fit model
    ab = AIVNAdaBoost()
    ab.fit(X_train, y_train, M = 50)

    # Predict on test set
    y_pred = ab.predict(X_test)
    print('The accuracy_score of the model is:', round(accuracy_score(y_test, y_pred), 4))

    spambase_data = dataset_manager.load_dataset('m3.adaboost.gradient.boost.20240913.spambase_data', read_file_options={'header': None})
    spambase_names = pd.DataFrame(dataset_manager.load_dataset('m3.adaboost.gradient.boost.20240913.spambase_names', read_file_options={'header': None, 'sep': ':', 'skiprows': range(0, 33)}))
    
    # df = pd.read_csv('/content/drive/MyDrive/AI2023/adaboost/spambase.data', header = None)
    # names = pd.read_csv('/content/drive/MyDrive/AI2023/adaboost/spambase.names', sep = ':', skiprows=range(0, 33), header = None)
    df = pd.DataFrame(spambase_data)
    col_names = list(spambase_names[0])
    col_names.append('Spam')
    df.columns = col_names
    print(df.head())

    df['Spam'] = df['Spam'] * 2 - 1

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'Spam').values, df['Spam'].values, test_size = 0.2, random_state = 2)

    # Fit model
    ab = AIVNAdaBoost()
    ab.fit(X_train, y_train, M = 50)

    # Predict on test set
    y_pred = ab.predict(X_test)
    print('The accuracy_score of the model is:', round(accuracy_score(y_test, y_pred), 4))

    # Using library
    ab_sk = AdaBoostClassifier(n_estimators = 50)
    ab_sk.fit(X_train, y_train)
    y_pred_sk = ab_sk.predict(X_test)
    print('The accuracy_score of the model is:', round(accuracy_score(y_test, y_pred_sk), 4))

if __name__ == "__main__":
    main()
