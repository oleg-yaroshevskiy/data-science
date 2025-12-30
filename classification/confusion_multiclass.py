import numpy as np
import matplotlib.pyplot as plt

# Custom implementations for multiclass classification metrics

def confusion_matrix_multiclass(y_true, y_pred, num_classes=None):
    """
    Build confusion matrix for multiclass classification.
    Matrix[i, j] = number of samples with true label i predicted as j
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm

def precision_per_class(cm):
    """
    Calculate precision for each class.
    Precision_i = TP_i / (TP_i + FP_i) = cm[i,i] / sum(cm[:, i])
    """
    precisions = []
    for i in range(len(cm)):
        tp = cm[i, i]
        tp_fp = np.sum(cm[:, i])  # All predicted as class i
        if tp_fp == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / tp_fp)
    return np.array(precisions)

def recall_per_class(cm):
    """
    Calculate recall for each class.
    Recall_i = TP_i / (TP_i + FN_i) = cm[i,i] / sum(cm[i, :])
    """
    recalls = []
    for i in range(len(cm)):
        tp = cm[i, i]
        tp_fn = np.sum(cm[i, :])  # All actual class i
        if tp_fn == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / tp_fn)
    return np.array(recalls)

def f1_per_class(cm):
    """
    Calculate F1 score for each class.
    F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
    """
    precisions = precision_per_class(cm)
    recalls = recall_per_class(cm)
    
    f1_scores = []
    for prec, rec in zip(precisions, recalls):
        if (prec + rec) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (prec * rec) / (prec + rec))
    
    return np.array(f1_scores)

def f1_macro(cm):
    """
    Macro-averaged F1 Score.
    Calculate F1 for each class, then take the simple average.
    Treats all classes equally regardless of support.
    """
    f1_scores = f1_per_class(cm)
    return np.mean(f1_scores)

def f1_micro(cm):
    """
    Micro-averaged F1 Score.
    Aggregate TP, FP, FN across all classes, then calculate F1.
    Weights each sample equally.
    
    For multiclass: Micro F1 = Micro Precision = Micro Recall = Accuracy
    """
    # Total true positives across all classes
    tp_total = np.sum(np.diag(cm))
    
    # Total predictions
    total = np.sum(cm)
    
    # For micro-averaging in multiclass:
    # Micro Precision = TP_total / Total_predictions = Accuracy
    # Micro Recall = TP_total / Total_actual = Accuracy
    # Therefore Micro F1 = Accuracy
    
    if total == 0:
        return 0.0
    
    return tp_total / total

def f1_weighted(cm):
    """
    Weighted F1 Score.
    Calculate F1 for each class, then take weighted average by support.
    Accounts for class imbalance.
    """
    f1_scores = f1_per_class(cm)
    
    # Support for each class (number of true instances)
    support = np.sum(cm, axis=1)
    
    # Weighted average
    total_support = np.sum(support)
    if total_support == 0:
        return 0.0
    
    weighted_f1 = np.sum(f1_scores * support) / total_support
    return weighted_f1

def accuracy_multiclass(cm):
    """
    Overall accuracy: correct predictions / total predictions
    """
    correct = np.sum(np.diag(cm))
    total = np.sum(cm)
    return correct / total if total > 0 else 0.0


# Generate synthetic multiclass data
print("=" * 70)
print("MULTICLASS CLASSIFICATION METRICS")
print("=" * 70)
print()

np.random.seed(42)
num_classes = 4
num_samples = 200

# Generate true labels with some class imbalance
class_probs = np.array([0.4, 0.3, 0.2, 0.1])  # Imbalanced classes
y_true = np.random.choice(num_classes, size=num_samples, p=class_probs)

# Generate predictions (with some confusion between classes)
y_pred = y_true.copy()
# Add some random misclassifications
num_errors = int(0.3 * num_samples)  # 30% error rate
error_indices = np.random.choice(num_samples, size=num_errors, replace=False)
for idx in error_indices:
    # Pick a different class
    other_classes = [c for c in range(num_classes) if c != y_true[idx]]
    y_pred[idx] = np.random.choice(other_classes)

print(f"Dataset: {num_samples} samples, {num_classes} classes")
print(f"Class distribution (true): {np.bincount(y_true)}")
print(f"Class distribution (pred): {np.bincount(y_pred)}")
print()

# Calculate confusion matrix
cm = confusion_matrix_multiclass(y_true, y_pred, num_classes=num_classes)

# Display Confusion Matrix
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print("Rows = True Labels, Columns = Predicted Labels")
print()

# Print header
print("True\\Pred", end="")
for j in range(num_classes):
    print(f"  Class {j}", end="")
print()

# Print matrix
for i in range(num_classes):
    print(f"Class {i}  ", end="")
    for j in range(num_classes):
        print(f"    {cm[i, j]:3d}", end="")
    print()
print()

# Calculate per-class metrics
precisions = precision_per_class(cm)
recalls = recall_per_class(cm)
f1_scores = f1_per_class(cm)
support = np.sum(cm, axis=1)

# Display per-class metrics
print("=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)
print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 70)

for i in range(num_classes):
    print(f"{i:<8} {precisions[i]:<12.4f} {recalls[i]:<12.4f} {f1_scores[i]:<12.4f} {support[i]:<10}")

print()

# Calculate and display aggregate metrics
print("=" * 70)
print("AGGREGATE METRICS")
print("=" * 70)

acc = accuracy_multiclass(cm)
macro_f1 = f1_macro(cm)
micro_f1 = f1_micro(cm)
weighted_f1 = f1_weighted(cm)

print(f"Accuracy:        {acc:.4f}")
print()
print(f"F1 Macro:        {macro_f1:.4f}")
print("  → Average F1 across all classes (treats all classes equally)")
print()
print(f"F1 Micro:        {micro_f1:.4f}")
print("  → Aggregate TP/FP/FN globally (weights each sample equally)")
print("  → For multiclass: Micro F1 = Accuracy")
print()
print(f"F1 Weighted:     {weighted_f1:.4f}")
print("  → Weighted average by support (accounts for class imbalance)")
print()

# Detailed explanation
print("=" * 70)
print("UNDERSTANDING MACRO vs MICRO vs WEIGHTED F1")
print("=" * 70)
print()
print("MACRO F1:")
print("  - Calculate F1 for each class independently")
print("  - Take simple average: (F1_0 + F1_1 + ... + F1_n) / n")
print("  - Treats all classes equally (good for balanced evaluation)")
print("  - Can be low if minority classes perform poorly")
print()
print("MICRO F1:")
print("  - Aggregate all TP, FP, FN across classes")
print("  - Calculate single F1 from aggregated counts")
print("  - Weights each sample equally (dominated by majority classes)")
print("  - In multiclass: Micro F1 = Micro Precision = Micro Recall = Accuracy")
print()
print("WEIGHTED F1:")
print("  - Calculate F1 for each class independently")
print("  - Take weighted average by class support (number of true samples)")
print("  - Accounts for class imbalance")
print("  - Better reflects overall model performance on imbalanced data")
print()

# Visualization 1: Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Multiclass Confusion Matrix', fontsize=16, fontweight='bold')
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [f'Class {i}' for i in range(num_classes)])
plt.yticks(tick_marks, [f'Class {i}' for i in range(num_classes)])

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')

plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass.png', dpi=300, bbox_inches='tight')
print("Confusion matrix heatmap saved as 'confusion_matrix_multiclass.png'")
print()

# Visualization 2: Per-Class Metrics Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(num_classes)
width = 0.6

# Plot 1: Precision, Recall, F1 per class
ax1 = axes[0]
x_pos = np.arange(num_classes)
width_bar = 0.25

ax1.bar(x_pos - width_bar, precisions, width_bar, label='Precision', color='skyblue')
ax1.bar(x_pos, recalls, width_bar, label='Recall', color='lightcoral')
ax1.bar(x_pos + width_bar, f1_scores, width_bar, label='F1-Score', color='lightgreen')

ax1.set_xlabel('Class', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'Class {i}' for i in range(num_classes)])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Plot 2: F1 Score Comparison
ax2 = axes[1]
f1_types = ['Macro', 'Micro', 'Weighted']
f1_values = [macro_f1, micro_f1, weighted_f1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = ax2.bar(f1_types, f1_values, color=colors, width=0.6)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.set_title('F1 Score Variants', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, f1_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Support per class
ax3 = axes[2]
ax3.bar(x, support, color='mediumpurple', width=0.6)
ax3.set_xlabel('Class', fontsize=11)
ax3.set_ylabel('Number of Samples', fontsize=11)
ax3.set_title('Class Support (Imbalance)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'Class {i}' for i in range(num_classes)])
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('multiclass_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Metrics comparison plot saved as 'multiclass_metrics_comparison.png'")
print()

# Visualization 3: Normalized Confusion Matrix (Percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('Normalized Confusion Matrix (Row-wise %)', fontsize=16, fontweight='bold')
plt.colorbar(label='Proportion')

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [f'Class {i}' for i in range(num_classes)])
plt.yticks(tick_marks, [f'Class {i}' for i in range(num_classes)])

# Add text annotations with percentages
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, f'{cm_normalized[i, j]:.2%}\n({cm[i, j]})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=10, fontweight='bold')

plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("Normalized confusion matrix saved as 'confusion_matrix_normalized.png'")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("All multiclass metrics have been calculated and displayed.")
print("Three PNG files have been generated:")
print("  1. confusion_matrix_multiclass.png - Confusion matrix heatmap")
print("  2. multiclass_metrics_comparison.png - Per-class metrics comparison")
print("  3. confusion_matrix_normalized.png - Normalized confusion matrix")
print("=" * 70)
