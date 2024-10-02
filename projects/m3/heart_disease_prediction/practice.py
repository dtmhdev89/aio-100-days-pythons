import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier

from sklearn import datasets

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
    sns.set_context("paper", font_scale = 1, rc = {"font.size": 3,"axes.titlesize": 15,"axes.labelsize": 10})

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='age', hue='target', order = df['age'].sort_values().unique())
    plt.xticks(np.arange(0, 80, 5))
    plt.title('Variation of Age for each target class')
    plt.show()

    # ax = sns.catplot(kind = 'count', data = df, x = 'age', hue = 'target', order = df['age'].sort_values().unique())
    # ax.ax.set_xticks(np.arange(0, 80, 5))
    # plt.title('Variation of Age for each target class')
    # plt.show()

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
    
    # Preprocessing
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # fit_transform
    # This method is applied to the training data (X_train).
    # It performs two operations:
    # Fit: The StandardScaler calculates the mean and standard deviation of the training data.
    # Transform: It then uses these statistics (mean and standard deviation) to scale the data by subtracting the mean and dividing by the standard deviation.
    # This ensures that the training data is standardized based on its own distribution.
    
    # transform
    # This method is applied to the test data (X_test).
    # It only transforms the data using the mean and standard deviation that were calculated from the training data during the fit process.
    # The test data is not refitted (i.e., the mean and standard deviation are not recalculated).
    # Instead, the scaling applied to the test data is consistent with the training data,
    # ensuring that the model's assumptions remain valid when applied to unseen data.

    # Some models (e.g., tree-based models like Decision Trees, Random Forest, or XGBoost) do not require feature scaling.
    # Tree-based models are generally invariant to the scaling of the input features, and scaling might even hurt performance
    # because these models rely on splitting based on feature values.
    # In contrast, models like SVM, KNN, and Logistic Regression are more sensitive to feature scales.
    # If you are using a tree-based model, applying StandardScaler might have unnecessarily transformed the data, causing a drop in accuracy.

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

    cls_lst = [('dtc', dtc), ('rfc', rfc), ('knn', knn), ('gc', gc), ('ad', ad), ('svc', svc)] 

    stacking_cls = StackingClassifier(estimators=cls_lst, final_estimator=xgb)
    stacking_cls.fit(X_train, y_train)
    predict_and_display_accurcy_for_train_and_test_data(X_train, X_test, y_train, y_test, stacking_cls)

    # Feature important
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


    clf = DecisionTreeClassifier(criterion='gini')
    # Fit the decision tree classifier
    clf = clf.fit(X_train, y_train)
    # Print the feature importances
    feature_importances = clf.feature_importances_

    # Sort the feature importances from greatest to least using the sorted indices
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_names = dataset.feature_names[sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    # Create a bar plot of the feature importances
    sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
    sns.barplot(x = sorted_importances, y = sorted_feature_names)
    plt.show()

if __name__ == "__main__":
    main()
