import numpy as np


def one_hot_to_labels(y):
    return np.argmax(y, axis=1)


y_true = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 1],
                   [0, 0, 1]])

y_pred = np.array([[0, 1, 1],
                   [0, 1, 1],
                   [0, 1, 0],
                   [0, 0, 0]])


# Accuracy
def accuracy(y_true, y_pred):
    y_true_labels = one_hot_to_labels(y_true)
    y_pred_labels = one_hot_to_labels(y_pred)
    correct = np.sum(y_true_labels == y_pred_labels)
    total = len(y_true_labels)
    return correct / total


print(one_hot_to_labels(y_true))
print(one_hot_to_labels(y_pred))

print(f"accuracy: {accuracy(y_true, y_pred):.2f}")


def precision(y_true, y_pred, average="micro"):
    y_true_labels = one_hot_to_labels(y_true)
    y_pred_labels = one_hot_to_labels(y_pred)

    classes = np.unique(y_true_labels)
    precision_per_class = []

    for cls in classes:
        tp = np.sum((y_pred_labels == cls) & (y_true_labels == cls))
        fp = np.sum((y_pred_labels == cls) & (y_true_labels != cls))
        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision_cls)

    if average == "micro":
        tp_total = np.sum((y_pred_labels == y_true_labels))
        fp_total = np.sum((y_pred_labels != y_true_labels) & (np.isin(y_pred_labels, classes)))
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    elif average == "macro":
        return np.mean(precision_per_class)
    elif average == "weighted":
        weights = [np.sum(y_true_labels == cls) for cls in classes]
        return np.average(precision_per_class, weights=weights)
    return 0
