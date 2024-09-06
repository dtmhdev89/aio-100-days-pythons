import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from dataset_man import DatasetManager

def main():
    # STEP 1: Load dataset and distribute data into input X and label Y
    iris_dataset = datasets.load_iris()
    print('====iris_dataset type')
    print(type(iris_dataset))
    iris_X = iris_dataset.data
    iris_Y = iris_dataset.target
    print(type(iris_X))
    print(type(iris_Y))

    print(pd.DataFrame(iris_X).info())
    print(pd.DataFrame(iris_X).describe())
    print(pd.DataFrame(iris_Y).info())
    print(pd.DataFrame(iris_Y))

    # STEP 1.1: Inspect data

    # STEP 2: Scalar (Transform, Vectorize,...)
    # No need to do it in this case

    # STEP 3: Split train, test data
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=50)

    print("train size: %d" %len(X_train))
    print("test size: %d" %len(X_test))

    # STEP 4: Model Instantiation
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=8, p=2)

    # STEP 5: Training. In this case maybe preparing the data for inference
    knn_clf.fit(X_train, y_train)

    # STEP 6: Predicting
    y_pred = knn_clf.predict(X_test)

    # STEP 7: Showing the result or plotting some kind of scores
    print ("Print results for 20 test data points:")
    print ("Predicted labels: ", y_pred[20:40])
    print ("Ground truth    : ", y_test[20:40])

    print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

    # -------------------------------------------------------
    print('-------Using with weight on distances')

    distance_weight_knn_clf = neighbors.KNeighborsClassifier(n_neighbors=8, p=2, weights='distance')
    distance_weight_knn_clf.fit(X_train, y_train)
    dw_y_pred = distance_weight_knn_clf.predict(X_test)

    print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, dw_y_pred)))
    print(classification_report(y_test, dw_y_pred))

    def my_weight(distances):
        sigma2 = 0.5 # can change this number

        return np.exp(-(distances**2/sigma2))
    
    with_weight_knn_clf = neighbors.KNeighborsClassifier(n_neighbors=8, p=2, weights=my_weight)
    with_weight_knn_clf.fit(X_train, y_train)
    w_y_pred = with_weight_knn_clf.predict(X_test)
    print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, w_y_pred)))
    print(classification_report(y_test, w_y_pred))

    dm = DatasetManager()
    print(dm.data_list)
    
    print('===========TeleCustomer Data')
    tele_df = dm.load_dataset('m3.knn.20240828.TeleCustomers', as_dataframe=True)
    print(tele_df.info())
    print(tele_df.head())
    print(tele_df.describe())

    X = tele_df.drop(['custcat'], axis=1)
    y = tele_df['custcat']

    # X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    X = preprocessing.StandardScaler().fit_transform(X.astype(float))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    tele_clf = neighbors.KNeighborsClassifier(n_neighbors=4)
    tele_clf.fit(X_train, y_train)

    y_pred = tele_clf.predict(X_test)

    print(classification_report(y_pred, y_test))

    error_rate = []
    acc_scores = []

    for i in range(1, 40):
        tele_knn = neighbors.KNeighborsClassifier(n_neighbors=i, weights='distance')
        tele_knn.fit(X_train, y_train)
        i_pred = tele_knn.predict(X_test)
        error_rate.append(np.mean(i_pred != y_test))
        acc_scores.append(accuracy_score(i_pred, y_test))

    # Rewrite above with pipeline
    print('======Write with pipeline')
    X1 = tele_df.drop(['custcat'], axis=1)
    y1 = tele_df['custcat']
    acc_2_scores = []

    knn_pl = Pipeline([
        ('scl', preprocessing.StandardScaler()),
        ('knn', neighbors.KNeighborsClassifier(weights='distance'))
    ])

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=4)

    for i in range(1, 40):
        knn_pl.set_params(knn__n_neighbors=i)
        knn_pl.fit(X1_train.astype(float), y1_train)
        acc_2_scores.append(knn_pl.score(X1_test.astype(float), y1_test))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed',
            marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40), acc_scores, color='blue', linestyle='dashed',
            marker='o', markerfacecolor='red', markersize=10)
    plt.title('Accuracy Score vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuary Score')
    print("Maximum accuracy:-",max(acc_scores),"at K =",acc_scores.index(max(acc_scores)))

    plt.plot(range(1,40), acc_2_scores, color='green', linestyle='dashed',
            marker='o', markerfacecolor='yellow', markersize=10)
    plt.title('Accuracy Score vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuary Score')
    print("Maximum accuracy:-",max(acc_2_scores),"at K =",acc_2_scores.index(max(acc_2_scores)))

    print('==========cruiseship')
    cruise_df = dm.load_dataset('m3.knn.20240828.cruise_ship_info', as_dataframe=True)
    print(cruise_df.head())
    print(cruise_df.info())

    pipe_lr = Pipeline([
        ('scl', preprocessing.StandardScaler()),
        ('slr', LinearRegression())])
    # knn_lr = KNeighborsRegressor(n_neighbors = 3)
    knn_lr = Pipeline([
        ('scl', preprocessing.StandardScaler()),
        ('slr', KNeighborsRegressor(n_neighbors=3))
    ])

    cols_selected = ['Tonnage', 'passengers', 'length', 'cabins','crew']
    X = cruise_df.loc[:, cols_selected[:-1]].values
    y = cruise_df[cols_selected]['crew'].values

    train_score_lr = []
    train_score_knn =  []
    n = 15
    sc_y = preprocessing.StandardScaler()
    
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
        train_score_lr = np.append(train_score_lr,
                                   np.mean(cross_val_score(pipe_lr,
                                   X_train, y_train_std,
                                   scoring ='r2' , cv = 10)))
        train_score_knn = np.append(train_score_knn,
                                    np.mean(cross_val_score(knn_lr,
                                    X_train, y_train_std,
                                    scoring ='r2' , cv = 10)))
    
    train_mean_lr = np.mean(train_score_lr)
    train_std_lr = np.std(train_score_lr)
    train_mean_knn = np.mean(train_score_knn)
    train_std_knn = np.std(train_score_knn)
    print('R2 train for lr: %.3f +/- %.3f' %(train_mean_lr, train_std_lr))
    print('R2 train for knn_lr: %.3f +/- %.3f' %(train_mean_knn, train_std_knn))

    plt.figure(figsize=(15,11))
    plt.plot(range(n),train_score_lr, color='blue', linestyle='dashed',
            marker='o', markerfacecolor='blue', markersize=10,
            label='linear regression')
    plt.fill_between(range(n),
                    train_score_lr + train_std_lr,
                    train_score_lr - train_std_lr,
                    alpha=0.15, color='blue')
    plt.plot(range(n),train_score_knn,color='green', linestyle='dashed',
            marker='s',markerfacecolor='green', markersize=10,
            label = 'Kneighbors regression')
    plt.fill_between(range(n),
                    train_score_knn + train_std_knn,
                    train_score_knn - train_std_knn,
                    alpha=0.15, color='green')
    plt.grid()
    plt.ylim(0.7,1)
    plt.title('Mean cross-validation R2 score vs. random state parameter', size = 14)
    plt.xlabel('Random state parameter', size = 14)
    plt.ylabel('Mean cross-validation R2 score', size = 14)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

