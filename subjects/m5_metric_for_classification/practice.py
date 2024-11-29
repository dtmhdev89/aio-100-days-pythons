from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\
    , precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

y_true = [1, 1, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted labels by the model
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Class_0", "Class_1"])
disp.plot(cmap='Blues')
plt.show()


# Compute precision
precision = precision_score(y_true, y_pred)

# Compute recall
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Compute F1 Score
f1 = f1_score(y_true, y_pred)

print(f"F1 Score: {f1:.2f}")
