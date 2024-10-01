import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier

from dataset_man import dataset_manager

def predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, model_instance_cls):
    model_name = model_instance_cls.__class__.__name__
    y_train_pred = model_instance_cls.predict(X_train)
    y_test_pred = model_instance_cls.predict(X_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for {} = {}'.format(model_name, accuracy_for_train))
    print('Accuracy for test set for {} = {}'.format(model_name, accuracy_for_test))

def main():
    cleveland_data = dataset_manager.load_dataset('m3.project.heart.disease.prediction.cleveland', read_file_options={'header': None})

    df = pd.DataFrame(cleveland_data)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.isna().sum())

    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    print(df.head())

    # bt1
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='age', hue='target')
    plt.xticks(['29', '39', '44', '49', '54', '59', '64', '69', '77'])
    plt.title('Variation of Age for each target class')
    plt.show()

    # bt2
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='sex', y='age', hue='target')
    plt.title('Distribution of age vs sex with target class')
    plt.show()

    # bt3 KNN
    test_size = 0.2
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    knn_cls = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
    knn_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, knn_cls)

    # bt4 SVM
    svc = SVC(kernel='rbf', random_state=42)
    svc.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, svc)

    # bt5 Naive Bayes
    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, nb_cls)

    # bt6 Decision Tree
    dct_cls = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2)
    dct_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, dct_cls)

    # bt7 Random Forest
    rf_cls = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2, n_estimators=10, random_state=42)
    rf_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, rf_cls)

    # bt8 AdaBoost
    adb_cls = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    adb_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, adb_cls)

    # bt9 Gredient Boost
    grad_cls = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, subsample=1.0, min_samples_split=2, max_depth=3, random_state=42)
    grad_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, grad_cls)

    # bt10 XGBoost
    xg_cls = XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
    xg_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, xg_cls)

    # bt11 Stacking
    dtc =  DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    knn =  KNeighborsClassifier()
    gc = GradientBoostingClassifier(random_state=42)
    svc = SVC(kernel = 'rbf', random_state=42)
    ad = AdaBoostClassifier(random_state=42)
    xgb = XGBClassifier()

    cls_lst = [('dtc', dtc), ('rfc', rfc), ('knn', knn), ('gc', gc), ('svc', svc), ('ad', ad)] 

    stacking_cls = StackingClassifier(estimators=cls_lst, final_estimator=xgb)
    stacking_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, stacking_cls)

if __name__ == "__main__":
    main()
