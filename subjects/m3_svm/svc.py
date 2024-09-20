import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset_man import dataset_manager

def perform_svc(X, y, transformer, non_rank_features, rank_features):
    X_transformed = transformer.fit_transform(X)
    print('X transformed: --\n', X_transformed[0])

    onehot_features = transformer.named_transformers_['OneHot'].get_feature_names_out(non_rank_features)
    rank_features_out = transformer.named_transformers_['Ordinal'].get_feature_names_out(rank_features)
    print('one hot features: --\n', onehot_features)
    print('rank features tranformed:--\n', rank_features_out)
    all_features = onehot_features.tolist() + rank_features

    X_encoded = pd.DataFrame(
        X_transformed,
        columns=all_features
    )

    print(X_encoded.head())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    normalizer = StandardScaler()
    X_normalized = normalizer.fit_transform(X_encoded)

    test_size = 0.3
    random_state = 1
    is_shuffle = True
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y_encoded,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of val samples: {X_val.shape[0]}')

    classifier = SVC(
        random_state=random_state
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_val)
    scores = accuracy_score(y_pred, y_val)

    print('Evaluation results on validation set:')
    print(f'Accuracy: {scores}')

def main():
    breast_cancer_data = dataset_manager.load_dataset('m3.svm.20240914.breast-cancer', read_file_options={'names': [
        'age',
        'meonpause',
        'tumor-size',
        'inv-nodes',
        'node-caps',
        'deg-malig',
        'breast',
        'breast-quad',
        'irradiat',
        'label'
    ]})

    df = pd.DataFrame(breast_cancer_data)
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.shape)
    print(df.isna().sum())

    # Filling missing data by mode of each columns
    df['node-caps'] = df['node-caps'].fillna(df['node-caps'].mode()[0])
    df['breast-quad'] = df['breast-quad'].fillna(df['breast-quad'].mode()[0])

    # most of columns are categorical type
    # Let's see unique values of each columns
    for col_name in df.columns:
        n_uniques = df[col_name].unique()
        print(f'Unique value in {col_name}: {n_uniques}')

    # Encode categorical features
    # Differ them by ranking and non-ranking features
    non_rank_features = ['meonpause', 'node-caps', 'breast', 'breast-quad', 'irradiat']
    rank_features = ['age', 'tumor-size', 'inv-nodes', 'deg-malig']

    y = df['label']
    X = df.drop('label', axis=1)

    transformer = ColumnTransformer(
        transformers=[
            ("OneHot", OneHotEncoder(drop='first'), non_rank_features),
            ("Ordinal", OrdinalEncoder(), rank_features)
        ],
        remainder='passthrough'
    )

    perform_svc(X, y, transformer, non_rank_features, rank_features)
    # X_transformed = transformer.fit_transform(X)
    # print('X transformed: --\n', X_transformed[0])

    # onehot_features = transformer.named_transformers_['OneHot'].get_feature_names_out(non_rank_features)
    # rank_features_out = transformer.named_transformers_['Ordinal'].get_feature_names_out(rank_features)
    # print('one hot features: --\n', onehot_features)
    # print('rank features tranformed:--\n', rank_features_out)
    # all_features = onehot_features.tolist() + rank_features

    # X_encoded = pd.DataFrame(
    #     X_transformed,
    #     columns=all_features
    # )

    # print(X_encoded.head())

    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)

    # normalizer = StandardScaler()
    # X_normalized = normalizer.fit_transform(X_encoded)

    # test_size = 0.3
    # random_state = 1
    # is_shuffle = True
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_normalized, y_encoded,
    #     test_size=test_size,
    #     random_state=random_state,
    #     shuffle=is_shuffle
    # )

    # print(f'Number of training samples: {X_train.shape[0]}')
    # print(f'Number of val samples: {X_val.shape[0]}')

    # classifier = SVC(
    #     random_state=random_state
    # )
    # classifier.fit(X_train, y_train)

    # y_pred = classifier.predict(X_val)
    # scores = accuracy_score(y_pred, y_val)

    # print('Evaluation results on validation set:')
    # print(f'Accuracy: {scores}')

    # Test without drop first
    transformer = ColumnTransformer(
        transformers=[
            ("OneHot", OneHotEncoder(), non_rank_features),
            ("Ordinal", OrdinalEncoder(), rank_features)
        ],
        remainder='passthrough'
    )

    perform_svc(X, y, transformer, non_rank_features, rank_features)

    # How to improve performance??


if __name__ == "__main__":
    main()
