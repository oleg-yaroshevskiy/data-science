import numpy as np
import matplotlib.pyplot as plt

# Custom implementations of sklearn metrics for interview preparation

def confusion_matrix(y_true, y_pred):
    """
    Build confusion matrix from scratch.
    Returns 2x2 matrix: [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate each component
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Build matrix
    cm = np.array([[tn, fp], [fn, tp]])
    return cm

def roc_curve(y_true, y_scores):
    """
    Calculate ROC curve from scratch.
    Returns: fpr, tpr, thresholds
    
    Algorithm:
    1. Sort scores in descending order
    2. For each unique threshold, calculate TPR and FPR
    3. TPR = TP / (TP + FN), FPR = FP / (FP + TN)
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Get unique thresholds (sorted descending) + add boundary points
    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]
    
    # Add boundary thresholds
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds, [thresholds[-1] - 1]])
    
    # Count total positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # Calculate TPR and FPR for each threshold
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        # Predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate TP, FP
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Calculate rates
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc(x, y):
    """
    Calculate Area Under Curve using trapezoidal rule.
    AUC = sum of trapezoid areas between consecutive points
    Trapezoid area = (x2 - x1) * (y1 + y2) / 2
    """
    x = np.array(x)
    y = np.array(y)
    
    # Sort by x values
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    # Calculate area using trapezoidal rule
    area = 0.0
    for i in range(len(x) - 1):
        # Width of trapezoid
        width = x[i + 1] - x[i]
        # Average height of trapezoid
        avg_height = (y[i] + y[i + 1]) / 2
        # Add trapezoid area
        area += width * avg_height
    
    return area

# Generate synthetic data
np.random.seed(42)
y_true = (np.random.rand(100) >= 0.2).astype(int)
y_pred_proba = np.random.beta(4, 2, size=100)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate confusion matrix components
tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fn = np.sum((y_true == 1) & (y_pred == 0))

print("=" * 60)
print("CONFUSION MATRIX COMPONENTS")
print("=" * 60)
print(f"True Positives (TP):  {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN):  {tn}")
print(f"False Negatives (FN): {fn}")
print()

# Display Confusion Matrix
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(y_true, y_pred)
print("                 Predicted")
print("              Negative  Positive")
print(f"Actual Negative  {cm[0, 0]:4d}      {cm[0, 1]:4d}")
print(f"Actual Positive  {cm[1, 0]:4d}      {cm[1, 1]:4d}")
print()

# Metric calculation functions
def accuracy(tp, fp, tn, fn):
    """Accuracy: (TP + TN) / Total"""
    return (tp + tn) / (tp + fp + tn + fn)

def precision(tp, fp, tn, fn):
    """Precision: TP / (TP + FP) - How many predicted positives are correct"""
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)

def recall(tp, fp, tn, fn):
    """Recall/Sensitivity: TP / (TP + FN) - How many actual positives are found"""
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def specificity(tp, fp, tn, fn):
    """Specificity: TN / (TN + FP) - How many actual negatives are correctly identified"""
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def f1_score(tp, fp, tn, fn):
    """F1 Score: Harmonic mean of precision and recall"""
    prec = precision(tp, fp, tn, fn)
    rec = recall(tp, fp, tn, fn)
    if (prec + rec) == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def f_beta_score(tp, fp, tn, fn, beta=2.0):
    """F-beta Score: Weighted harmonic mean of precision and recall
    beta > 1: emphasizes recall
    beta < 1: emphasizes precision
    """
    prec = precision(tp, fp, tn, fn)
    rec = recall(tp, fp, tn, fn)
    if (beta**2 * prec + rec) == 0:
        return 0.0
    return (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)

# Calculate and print all metrics
print("=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)
acc = accuracy(tp, fp, tn, fn)
prec = precision(tp, fp, tn, fn)
rec = recall(tp, fp, tn, fn)
spec = specificity(tp, fp, tn, fn)
f1 = f1_score(tp, fp, tn, fn)
f2 = f_beta_score(tp, fp, tn, fn, beta=2.0)
f05 = f_beta_score(tp, fp, tn, fn, beta=0.5)

print(f"Accuracy:           {acc:.4f}")
print(f"Precision:          {prec:.4f}")
print(f"Recall/Sensitivity: {rec:.4f}")
print(f"Specificity:        {spec:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"F2 Score (β=2):     {f2:.4f}  (emphasizes recall)")
print(f"F0.5 Score (β=0.5): {f05:.4f}  (emphasizes precision)")
print()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("=" * 60)
print("ROC AUC SCORE")
print("=" * 60)
print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")
print()

# Plot ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC curve saved as 'roc_curve.png'")
print()

# Additional visualization: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, fontweight='bold')

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix heatmap saved as 'confusion_matrix.png'")
print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("All metrics have been calculated and displayed.")
print("Two PNG files have been generated:")
print("  1. roc_curve.png - ROC curve visualization")
print("  2. confusion_matrix.png - Confusion matrix heatmap")
print("=" * 60)