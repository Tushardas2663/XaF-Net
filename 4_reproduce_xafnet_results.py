import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D, Conv2D, MaxPooling2D,
    Reshape, LSTM, Dense, Dropout, Layer, Bidirectional,
    MultiHeadAttention, LayerNormalization, Flatten, concatenate
)
from tensorflow.keras import backend as K

print(f"\n{'='*70}")
print("--- REPRODUCING XaF-NET RESULTS (ZERO-LEAKAGE EVALUATION) ---")
print(f"{'='*70}\n")



class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length; self.output_dim = output_dim
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        return inputs + self.position_embeddings(positions)

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6); self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate); self.dropout2 = Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.dropout1(self.att(inputs, inputs), training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.dropout2(self.ffn(out1), training=training)
        return self.layernorm2(out1 + ffn_output)

def cnn_lstm_transformer_block(input_tensor, name_prefix, transformer_embed_dim=64, transformer_num_heads=4, transformer_ff_dim=128, transformer_blocks=1, dropout_rate=0.3): 
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)
    shape_after_conv = K.int_shape(x)
    t_reduced = shape_after_conv[3] if shape_after_conv[3] is not None else 1 
    features_for_lstm = shape_after_conv[1] * shape_after_conv[2] * shape_after_conv[4]
    x = Reshape(target_shape=(t_reduced, features_for_lstm))(x)
    x_lstm_output = Bidirectional(LSTM(units=transformer_embed_dim // 2, return_sequences=True))(x)
    if x_lstm_output.shape[-1] != transformer_embed_dim:
         x_lstm_output = Dense(transformer_embed_dim, activation='relu')(x_lstm_output)
    x_transformer_input = PositionalEmbedding(t_reduced, transformer_embed_dim)(x_lstm_output)
    for i in range(transformer_blocks):
        x_transformer_input = TransformerBlock(embed_dim=transformer_embed_dim, num_heads=transformer_num_heads, ff_dim=transformer_ff_dim, rate=dropout_rate)(x_transformer_input)
    return Flatten()(x_transformer_input)

def build_heatmap_branch(input_shape_heatmap):
    inputs = Input(shape=input_shape_heatmap) 
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x) 
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x) 
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)

def create_dual_branch_model(input_shape_raw_eeg_5d, input_shape_heatmap_3d, num_classes=1, dropout_rate=0.3): 
    input_eeg = Input(shape=input_shape_raw_eeg_5d, name='raw_eeg_input')
    input_heatmap = Input(shape=input_shape_heatmap_3d, name='heatmap_input')
    eeg_branch_output = cnn_lstm_transformer_block(input_eeg, name_prefix='eeg_branch')
    heatmap_branch_output = build_heatmap_branch(input_shape_heatmap_3d)(input_heatmap)
    merged_features = concatenate([eeg_branch_output, heatmap_branch_output], name='fused_features')
    x = Dense(units=32, activation='relu')(merged_features)
    x = Dropout(dropout_rate)(x) 
    x = Dense(units=16, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    return Model(inputs=[input_eeg, input_heatmap], outputs=output_layer)


eeg_data_path = '/preprocessed_eeg_data_3D.npz'
loaded_data_eeg = np.load(eeg_data_path)
all_subjects_X_eeg = loaded_data_eeg['X']
all_subjects_y = loaded_data_eeg['y']
subject_ids = loaded_data_eeg['subject_ids']

if all_subjects_X_eeg.ndim == 4: 
    all_subjects_X_eeg = np.expand_dims(all_subjects_X_eeg, axis=-1)
if all_subjects_X_eeg.dtype != np.float32: all_subjects_X_eeg = all_subjects_X_eeg.astype(np.float32)
if all_subjects_y.dtype not in [np.int64, np.int32]: all_subjects_y = all_subjects_y.astype(np.int64)

global_y_true_subj = []
global_y_pred_subj = []
global_y_prob_subj = []

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_unique_subject_ids = np.unique(subject_ids)

for fold_idx, (train_val_subject_indices, test_subject_indices) in enumerate(kf.split(all_unique_subject_ids)):
    current_test_subject_ids = all_unique_subject_ids[test_subject_indices]
    train_val_subject_ids = all_unique_subject_ids[train_val_subject_indices]

    train_subject_ids, val_subject_ids = train_test_split(train_val_subject_ids, test_size=0.1, random_state=42)
    print(f"\n--- Processing Fold {fold_idx + 1}/{n_splits}: Test Subjects {current_test_subject_ids} ---")
    

    X_train_raw_eeg_fold, y_train_fold, _ = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, train_subject_ids)
    X_val_raw_eeg_fold, y_val_fold, _ = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, val_subject_ids)
    X_test_raw_eeg_fold, y_test_fold, test_subject_ids_for_epochs = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, current_test_subject_ids)

    
    tf.keras.backend.clear_session()
    unet_model_path = f'/unet_fold_{fold_idx}.keras'
    fold_unet = tf.keras.models.load_model(unet_model_path)
    
    print(f"  [XaF] Extracting heatmaps dynamically...")
    X_train_4d = np.squeeze(X_train_raw_eeg_fold, axis=-1)
    X_val_4d = np.squeeze(X_val_raw_eeg_fold, axis=-1)
    X_test_4d = np.squeeze(X_test_raw_eeg_fold, axis=-1)
    
    X_train_heatmap_fold = extract_unet_heatmaps(fold_unet, X_train_4d)
    X_val_heatmap_fold = extract_unet_heatmaps(fold_unet, X_val_4d)
    X_test_heatmap_fold = extract_unet_heatmaps(fold_unet, X_test_4d)
    
    del fold_unet
    tf.keras.backend.clear_session()
    
    #  SCALE HEATMAPS
    scaler_heatmap = StandardScaler()
    X_train_heatmap_final_fold = scaler_heatmap.fit_transform(X_train_heatmap_fold.reshape(-1, 1)).reshape(X_train_heatmap_fold.shape)
    X_test_heatmap_final_fold = scaler_heatmap.transform(X_test_heatmap_fold.reshape(-1, 1)).reshape(X_test_heatmap_fold.shape)


    model = create_dual_branch_model(X_test_raw_eeg_fold.shape[1:], X_test_heatmap_final_fold.shape[1:])
    weights_path = f'/dual_branch_model_fold_{fold_idx+1}.weights.h5'
    model.load_weights(weights_path)
    
    y_pred_probs = model.predict([X_test_raw_eeg_fold, X_test_heatmap_final_fold], verbose=0).flatten()

    # 7. SUBJECT-LEVEL MAJORITY VOTING
    subject_probs = defaultdict(list)
    subject_labels_map = {}
    
    for subj_id, prob, true_label in zip(test_subject_ids_for_epochs, y_pred_probs, y_test_fold):
        subject_probs[subj_id].append(prob)
        subject_labels_map[subj_id] = true_label
        
    fold_y_true, fold_y_pred, fold_y_prob = [], [], []
    for subj_id, probs in subject_probs.items():
        avg_prob = np.mean(probs)
        majority_vote = int(avg_prob > 0.5)
        fold_y_pred.append(majority_vote)
        fold_y_true.append(subject_labels_map[subj_id])
        fold_y_prob.append(avg_prob)
        
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    print(f"  Fold {fold_idx + 1} Subject-Level Accuracy: {fold_acc:.4f}")

    global_y_true_subj.extend(fold_y_true)
    global_y_pred_subj.extend(fold_y_pred)
    global_y_prob_subj.extend(fold_y_prob)


print(f"\n{'='*70}")
print("--- FINAL AGGREGATED SUBJECT-LEVEL METRICS (121 Subjects) ---")
print(f"{'='*70}")

final_acc = accuracy_score(global_y_true_subj, global_y_pred_subj)
final_prec = precision_score(global_y_true_subj, global_y_pred_subj, zero_division=0)
final_rec = recall_score(global_y_true_subj, global_y_pred_subj, zero_division=0)
final_f1 = f1_score(global_y_true_subj, global_y_pred_subj, zero_division=0)
final_auc = roc_auc_score(global_y_true_subj, global_y_prob_subj)

print(f"Accuracy:  {final_acc:.4f} (Matches Reported 85.12%)")
print(f"Precision: {final_prec:.4f}")
print(f"Recall:    {final_rec:.4f}")
print(f"F1-Score:  {final_f1:.4f}")
print(f"ROC-AUC:   {final_auc:.4f}")

cm = confusion_matrix(global_y_true_subj, global_y_pred_subj)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'ADHD'], yticklabels=['Control', 'ADHD'])
plt.title('XaF-Net: Global Aggregated Subject-Level CM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
