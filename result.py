# Ground truth labels (from the nutritionist)
y_true = ['MS','NS','MS','MS','S','S','MS','MS','MS','MS','S','MS','MS','S','MS','MS','MS','MS','S','NS','S','MS','S','MS','S','NS','MS','NS','S','S','MS','S','MS','S','NS','MS','S','NS','MS','NS','S','S','MS','MS','NS','MS','S','MS','S','MS','S','S','S','S','S','S','S']

# Your model’s predicted labels
y_pred = ['MS','NS','MS','MS','S','S','MS','MS','MS','MS','S','MS','MS','S','MS','MS','NS','S','S','NS','S','MS','S','MS','S','NS','MS','NS','S','S','MS','S','MS','S','NS','MS','S','NS','MS','NS','S','S','MS','MS','NS','MS','S','MS','S','MS','S','S','S','S','S','S','S']

from sklearn.metrics import f1_score, classification_report, accuracy_score

# F1 score for each class equally (macro)
f1_macro = f1_score(y_true, y_pred, average='macro')

# Weighted F1 (accounts for class imbalance)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Detailed report (includes precision, recall, F1 for each class)
report = classification_report(y_true, y_pred)

print("F1 Score (Macro):", f1_macro)
print("F1 Score (Weighted):", f1_weighted)
print("\nDetailed Classification Report:\n", report)

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)