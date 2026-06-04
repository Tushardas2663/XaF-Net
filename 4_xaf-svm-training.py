import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""After running the xaf feature extraction script for 5 times for the 5 folds, run this script to reproduce the exact xaf-svm results. the optimal svm params shown below
are found after a grid search using strictly the training data. the grid search code is provided at the end of the script as a doc string for reference."""

print(f"\n{'='*60}")
print("--- SUBJECT-LEVEL SVM EVALUATION ON XaF (5-FOLD CV) ---")
print(f"{'='*60}\n")


optimal_svm_params = {
    0: {'C': 0.420557, 'gamma': 0.050477, 'kernel': 'rbf'},
    1: {'C': 0.420557, 'gamma': 0.050477, 'kernel': 'rbf'},
    2: {'C': 21.830968, 'gamma': 0.061737, 'kernel': 'poly'},
    3: {'C': 9.717775, 'gamma': 0.008612, 'kernel': 'poly'},
    4: {'C': 0.137832, 'gamma': 0.066471, 'kernel': 'rbf'}
}


fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

global_y_true_subj = []
global_y_pred_subj = []

# 3. Loop through all 5 folds
for fold in range(5):
    print(f"\n--- PROCESSING FOLD {fold} ---")
    
   
    filepath = f'/fold_{fold}_xaf_data.npz'
    data = np.load(filepath)
    
    X_train = np.squeeze(data['X_heatmap_train'])
    y_train = data['y_train']
    
    X_test = np.squeeze(data['X_heatmap_test'])
    y_test = data['y_test']
    subj_test = data['subj_test'] # Crucial for subject-level voting
    
    # Flatten heatmaps for SVM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Initialize and Train SVM with Optimal Params
    params = optimal_svm_params[fold]
    svm = SVC(C=params['C'], gamma=params['gamma'], kernel=params['kernel'], 
              class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
  
    y_pred_epoch = svm.predict(X_test_scaled)
    
    # ==========================================
    # AGGREGATE TO SUBJECT LEVEL (MAJORITY VOTE)
    # ==========================================
    subject_preds = defaultdict(list)
    subject_labels = {}
    
    for subj_id, pred, true_label in zip(subj_test, y_pred_epoch, y_test):
        subject_preds[subj_id].append(pred)
        subject_labels[subj_id] = true_label
        
    fold_y_true_subj = []
    fold_y_pred_subj = []
    
    for subj_id, preds in subject_preds.items():
        majority_vote = int(np.mean(preds) > 0.5)
        fold_y_pred_subj.append(majority_vote)
        fold_y_true_subj.append(subject_labels[subj_id])
        
    
    acc = accuracy_score(fold_y_true_subj, fold_y_pred_subj)
    prec = precision_score(fold_y_true_subj, fold_y_pred_subj, zero_division=0)
    rec = recall_score(fold_y_true_subj, fold_y_pred_subj, zero_division=0)
    f1 = f1_score(fold_y_true_subj, fold_y_pred_subj, zero_division=0)
    
    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)
    
    global_y_true_subj.extend(fold_y_true_subj)
    global_y_pred_subj.extend(fold_y_pred_subj)
    
    print(f"Subject Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    
   
    cm = confusion_matrix(fold_y_true_subj, fold_y_pred_subj)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Fold {fold}: Subject-Level SVM Confusion Matrix")
    plt.ylabel('True Class'); plt.xlabel('Predicted Class')
    plt.show()


print(f"\n{'='*60}")
print("--- 5-FOLD SUBJECT-LEVEL CROSS-VALIDATION SUMMARY (SVM) ---")
print(f"{'='*60}")

print(f"Accuracy:  {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Recall:    {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"F1-Score:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")


global_cm = confusion_matrix(global_y_true_subj, global_y_pred_subj)
plt.figure(figsize=(6, 5))
sns.heatmap(global_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
#plt.title(f"Global 5-Fold Aggregated Confusion Matrix\n(Subject-Level SVM, Total N={len(global_y_true_subj)})", fontsize=12, fontweight='bold')
plt.ylabel('True Class', fontsize=12); plt.xlabel('Predicted Class', fontsize=12)
plt.savefig('Global_SVM_Subject_Confusion_Matrix.pdf', dpi=300)
plt.show()

print("\n Evaluation Complete! You can now copy these metrics directly into your manuscript.")


# below is the grid search guidelines. do the grid search separately for each fold to get fold wise optimal params. strictly use train data to avoid leakage.

"""
X_train_flat = X_heatmap_train.reshape(X_heatmap_train.shape[0], -1)
X_test_flat = X_heatmap_test.reshape(X_heatmap_test.shape[0], -1)


svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', random_state=42, probability=False)) 
])

svm_param_grid = {
    'svm__C': loguniform(0.1, 100),       
    'svm__gamma': loguniform(0.001, 1),  
    'svm__kernel': ['rbf', 'poly', 'sigmoid'] 
}


print("Tuning SVM (Hunting for optimal C and gamma)...")
svm_search = RandomizedSearchCV(estimator=svm_pipeline, param_distributions=svm_param_grid, n_iter=20,  scoring='accuracy', 
    cv=3, verbose=1, random_state=42,n_jobs=-1
)"""

svm_search.fit(X_train_flat, y_train)
best_svm = svm_search.best_estimator_
