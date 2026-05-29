import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"\n{'='*50}")
print(f"--- ABLATION STUDY: RAW EEG PERFORMANCE (FOLD {TARGET_FOLD}) ---")
print(f"{'='*50}\n")
#load the model and data above.
# ==========================================
# 1. FLATTEN RAW EEG FOR CLASSICAL ML
# ==========================================
print("Flattening Raw EEG for ML Models...")
# (N, 8, 8, 640) -> (N, 40960)
X_train_raw_flat = X_train_eeg.reshape(X_train_eeg.shape[0], -1)
X_test_raw_flat = X_test_eeg.reshape(X_test_eeg.shape[0], -1)

print(f"Flattened Raw Train Shape: {X_train_raw_flat.shape}")
print(f"Flattened Raw Test Shape:  {X_test_raw_flat.shape}")

# ==========================================
# 2. CLASSICAL ML MODELS EVALUATION (RAW EEG)
# ==========================================
print("\n--- Evaluating Classical ML Models on RAW EEG ---")

ml_models = {
    #'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1,verbose=False)
    #'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
}

for name, model in ml_models.items():
    print(f"Training {name}... (This may take a moment due to 40k+ features)")
    model.fit(X_train_raw_flat, y_train)
    y_pred = model.predict(X_test_raw_flat)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"{name:>13} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# ==========================================
# 3. DEEP LEARNING EVALUATION (RAW EEG)
# ==========================================
print("\n--- Evaluating Simple CNN on RAW EEG ---")

def build_raw_eeg_classifier(input_shape):
    """Same architecture as heatmap classifier, but accepts 640 channels"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

tf.keras.backend.clear_session()
# Input shape is now (8, 8, 640)
raw_cnn_classifier = build_raw_eeg_classifier(input_shape=(X_train_eeg.shape[1], X_train_eeg.shape[2], X_train_eeg.shape[3]))
raw_cnn_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision')])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("Training Simple CNN on Raw EEG...")
raw_cnn_classifier.fit(
    X_train_eeg, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_val_eeg, y_val), # Using the explicit validation set we created
    callbacks=[early_stopping], 
    verbose=0 
)

# Evaluate on the Test set
eval_results = raw_cnn_classifier.evaluate(X_test_eeg, y_test, verbose=0)
cnn_loss, cnn_acc, cnn_prec = eval_results[0], eval_results[1], eval_results[2]

y_pred_prob_cnn = raw_cnn_classifier.predict(X_test_eeg, verbose=0)
y_pred_cnn = (y_pred_prob_cnn > 0.5).astype(int).flatten()
cnn_rec = recall_score(y_test, y_pred_cnn, zero_division=0)
cnn_f1 = f1_score(y_test, y_pred_cnn, zero_division=0)

print(f"{'Simple CNN':>13} -> Acc: {cnn_acc:.4f} | Prec: {cnn_prec:.4f} | Rec: {cnn_rec:.4f} | F1: {cnn_f1:.4f}")
print("\n--- Ablation Study Complete ---")
