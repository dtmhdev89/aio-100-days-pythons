import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from dataset_man import dataset_manager

def main():
    dm = dataset_manager
    credit_card_data = dm.load_gdrive_dataset('11EBwEw6yNRWgNzi7XiLXABpfYG_q7iHY')
    creditdata_df = pd.DataFrame(credit_card_data)
    print(creditdata_df.head())
    print(creditdata_df.columns)

    false_cls = creditdata_df[creditdata_df['Class']==1]
    true_cls = creditdata_df[creditdata_df['Class']==0]
    n = len(false_cls)/float(len(true_cls))
    print(n)
    print('False Detection : {}'.format(len(creditdata_df[creditdata_df['Class']==1])))
    print('True Detection:{}'.format(len(creditdata_df[creditdata_df['Class']==0])),"\n")

    X = creditdata_df.drop('Class', axis=1)
    y = creditdata_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtree_cls_pl = Pipeline([
        ('cls', DecisionTreeClassifier())
    ])

    dtree_cls_pl.fit(X_train, y_train)

    y_pred = dtree_cls_pl.predict(X_test)

    print(y_pred[:10])
    print(y_test[:10].values)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy:", accuracy)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat) # [[tn, fp], [fn, tp]]

    precision=precision_score(y_test, y_pred, pos_label=1)*100
    print('\n Score Precision :\n',precision )

    #Recall
    # Recall = TP / (TP + FN)
    recall=recall_score(y_test, y_pred, pos_label=1)*100
    print("\n Recall Score :\n", recall)

    fscore=f1_score(y_test, y_pred, pos_label=1)*100
    print("\n F1 Score :\n", fscore)

if __name__ == "__main__":
    main()
