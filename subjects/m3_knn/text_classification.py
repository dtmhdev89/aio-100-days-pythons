import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def main():
    corpus = [
        "góp gió gặt bão",
        "có làm mới có ăn",
        "đất lành chim đậu",
        "ăn cháo đá bát",
        "gậy ông đập lưng ông",
        "qua cầu rút ván"
    ]

    labels = [1, 1, 1, 0, 0, 0] # 1: positive - 0: negative

    cate_2_label = {
        "positive": 1,
        "negative": 0
    }

    X = np.array(corpus)
    y = np.array(labels)

    def label_2_cate(labels):
        key_list = list(cate_2_label.keys())
        val_list = list(cate_2_label.values())

        position = [val_list.index(label) for label in labels]
    
        return np.array(key_list)[position]

    text_clf_model = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', KNeighborsClassifier(n_neighbors=1)),
                        ])
    
    text_clf_model.fit(X, y)
    preds = text_clf_model.predict(X)
    print(preds)

    # inference
    test_text = np.array(["không làm cạp đất mà ăn"])

    y_pred = text_clf_model.predict(test_text)
    print(y_pred)

    print(label_2_cate(y_pred))

if __name__ == "__main__":
    main()
