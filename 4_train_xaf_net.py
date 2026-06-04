import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import pandas as pd # Added for subject voting

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D, Conv2D,MaxPooling2D, # Added Conv2D
    Reshape, LSTM, Dense, Dropout, Layer, Bidirectional,
    MultiHeadAttention, LayerNormalization, Add, Flatten, Lambda, concatenate # Added concatenate
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


eeg_data_path = '/preprocessed_eeg_data_3D.npz' 

if os.path.exists(eeg_data_path):
    print(f"Loading processed EEG data from {eeg_data_path}")
    loaded_data_eeg = np.load(eeg_data_path)
    all_subjects_X_eeg = loaded_data_eeg['X']
    all_subjects_y = loaded_data_eeg['y']
    subject_ids = loaded_data_eeg['subject_ids']
    
    print("EEG data loaded successfully.")
    print("Shape of all_subjects_X_eeg:", all_subjects_X_eeg.shape)
    
 
    if all_subjects_X_eeg.ndim == 4: # If (N, H, W, T)
        all_subjects_X_eeg = np.expand_dims(all_subjects_X_eeg, axis=-1)
        print(f"Reshaped EEG data to: {all_subjects_X_eeg.shape}")
    if all_subjects_X_eeg.dtype != np.float32: all_subjects_X_eeg = all_subjects_X_eeg.astype(np.float32)
    if all_subjects_y.dtype not in [np.int64, np.int32]: all_subjects_y = all_subjects_y.astype(np.int64)

else:
    print(f"Error: EEG data file not found at {eeg_data_path}"); exit()


grid_height = all_subjects_X_eeg.shape[1]
grid_width = all_subjects_X_eeg.shape[2]
n_timepoints_original_eeg = all_subjects_X_eeg.shape[3]
input_shape_raw_eeg = (grid_height, grid_width, n_timepoints_original_eeg, 1) # 5D
input_shape_heatmap = (8, 8, 1)


early_stopping_callback = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=3,
    verbose=1, mode='max', restore_best_weights=False
)



def get_data_for_subjects(data_array, labels_array, subject_ids_array, list_of_subject_ids_for_split):
    labels_array=np.array(labels_array)
    indices_for_split = np.isin(subject_ids_array, list_of_subject_ids_for_split)
    
    if isinstance(data_array, list) or isinstance(data_array, tuple):
       
        filtered_data = [arr[indices_for_split] for arr in data_array]
    else:
        
        filtered_data = data_array[indices_for_split]
        
    return filtered_data, labels_array[indices_for_split], subject_ids_array[indices_for_split]


class PositionalEmbedding(Layer):
   
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length; self.output_dim = output_dim
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.supports_masking = True
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions); return inputs + embedded_positions
    def compute_mask(self, inputs, mask=None): return mask
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"sequence_length": self.sequence_length,"output_dim": self.output_dim}); return config


class TransformerBlock(Layer):
   
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.ff_dim = ff_dim; self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6); self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate); self.dropout2 = Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs); attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output); ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training); return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}); return config


def cnn_lstm_transformer_block(input_tensor, name_prefix, # This is input_tensor_5D
                               transformer_embed_dim=64, 
                               transformer_num_heads=4,
                               transformer_ff_dim=128,
                               transformer_blocks=1,
                               dropout_rate=0.3): 
   
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name=f'{name_prefix}_conv3d_1')(input_tensor)
    x = BatchNormalization(name=f'{name_prefix}_bn_1')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name=f'{name_prefix}_pool_1')(x)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name=f'{name_prefix}_conv3d_2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn_2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name=f'{name_prefix}_pool_2')(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', name=f'{name_prefix}_conv3d_3')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn_3')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name=f'{name_prefix}_pool_3')(x)
    shape_after_conv = K.int_shape(x)
    t_reduced = shape_after_conv[3] if shape_after_conv[3] is not None else -1
    if t_reduced == 0: t_reduced = 1 
    features_for_lstm = shape_after_conv[1] * shape_after_conv[2] * shape_after_conv[4]
    x = Reshape(target_shape=(t_reduced, features_for_lstm), name=f'{name_prefix}_reshape')(x)
    x_lstm_output = Bidirectional(LSTM(units=transformer_embed_dim // 2, return_sequences=True), name=f'{name_prefix}_bilstm_1')(x)
    if x_lstm_output.shape[-1] != transformer_embed_dim:
         x_lstm_output = Dense(transformer_embed_dim, activation='relu', name=f'{name_prefix}_lstm_output_dense')(x_lstm_output)
    x_transformer_input = PositionalEmbedding(t_reduced, transformer_embed_dim, name=f'{name_prefix}_positional_embedding')(x_lstm_output)
    for i in range(transformer_blocks):
        x_transformer_input = TransformerBlock(
            embed_dim=transformer_embed_dim, num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim, rate=dropout_rate, name=f'{name_prefix}_transformer_block_{i+1}'
        )(x_transformer_input)
    return Flatten(name=f'{name_prefix}_flatten_transformer_output')(x_transformer_input)


def build_heatmap_branch(input_shape_heatmap, name_prefix='heatmap_branch'):
    """Builds a simple CNN branch to extract features from 8x8 heatmaps."""
    inputs = Input(shape=input_shape_heatmap, name=f'{name_prefix}_input') # (8, 8, 1)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name=f'{name_prefix}_conv1')(inputs)
    x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(x) # 4x4
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name=f'{name_prefix}_conv2')(x)
    x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(x) # 2x2
    x = Flatten(name=f'{name_prefix}_flatten')(x)
    # Output features (e.g., shape (None, 2*2*32 = 128))
    return Model(inputs=inputs, outputs=x, name=name_prefix)


def create_dual_branch_model(input_shape_raw_eeg_5d, input_shape_heatmap_3d, num_classes=1,
                             eeg_transformer_embed_dim=64,
                             eeg_transformer_num_heads=4,
                             eeg_transformer_ff_dim=128,
                             eeg_transformer_blocks=1,
                             dropout_rate=0.3): # Your DSAEN dropout rate

    input_eeg = Input(shape=input_shape_raw_eeg_5d, name='raw_eeg_input')
    input_heatmap = Input(shape=input_shape_heatmap_3d, name='heatmap_input')

    
    eeg_branch_output = cnn_lstm_transformer_block(
        input_eeg,
        name_prefix='eeg_branch',
        transformer_embed_dim=eeg_transformer_embed_dim,
        transformer_num_heads=eeg_transformer_num_heads,
        transformer_ff_dim=eeg_transformer_ff_dim,
        transformer_blocks=eeg_transformer_blocks,
        dropout_rate=0.3 # Dropout within the block
    )

 
    heatmap_branch = build_heatmap_branch(input_shape_heatmap_3d)
    heatmap_branch_output = heatmap_branch(input_heatmap)

   
    merged_features = concatenate([eeg_branch_output, heatmap_branch_output], name='fused_features_after_branches')

  
    x = Dense(units=32, activation='relu', name='dense_fusion_1')(merged_features)
    x = Dropout(dropout_rate)(x) 
    x = Dense(units=16, activation='relu', name='dense_fusion_2')(x)
    x = Dropout(dropout_rate)(x)

    if num_classes == 1: # Binary classification
        output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    else: # Multi-class classification
        output_layer = Dense(units=num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input_eeg, input_heatmap], outputs=output_layer)
    return model


print("\n--- Starting 5-Fold Subject-Wise Cross-Validation (Dual Branch) ---")


fold_accuracies = []
fold_losses = []
fold_predictions_per_fold = [] 
fold_true_labels_per_fold = [] 
fold_subject_ids_per_fold = [] 
subject_level_results = [] 


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_unique_subject_ids = np.unique(subject_ids)
all_subjects_y = np.array(all_subjects_y)
subject_labels = [all_subjects_y[list(subject_ids).index(sid)] for sid in all_unique_subject_ids]


for fold_idx, (train_val_subject_indices, test_subject_indices) in enumerate(kf.split(all_unique_subject_ids)):

    current_test_subject_ids = all_unique_subject_ids[test_subject_indices]
    train_val_subject_ids = all_unique_subject_ids[train_val_subject_indices]

  
    subject_to_label = {sid: label for sid, label in zip(all_unique_subject_ids, subject_labels)}
    subject_labels_for_split = np.array([subject_to_label[sid] for sid in train_val_subject_ids])
    train_subject_ids, val_subject_ids = train_test_split(
        train_val_subject_ids,
        test_size=0.1, 
        random_state=42,
    
    )
    print("Validation subjects ", val_subject_ids)
    print(f"\n--- Processing Fold {fold_idx + 1}/{n_splits}: Test Subjects {current_test_subject_ids} ---")
    
   
    X_train_raw_eeg_fold, y_train_fold, _ = get_data_for_subjects(
        all_subjects_X_eeg, all_subjects_y, subject_ids, train_subject_ids
    )
    X_val_raw_eeg_fold, y_val_fold, _ = get_data_for_subjects(
        all_subjects_X_eeg, all_subjects_y, subject_ids, val_subject_ids
    )
    X_test_raw_eeg_fold, y_test_fold, test_subject_ids_for_epochs = get_data_for_subjects(
        all_subjects_X_eeg, all_subjects_y, subject_ids, current_test_subject_ids
    )

    print(f"  Fold {fold_idx+1} - Train: {y_train_fold.shape}, Val: {y_val_fold.shape}, Test: {y_test_fold.shape}")


    print(f"  [XaF] Loading strict Fold {fold_idx + 1} U-Net to prevent leakage...")
    tf.keras.backend.clear_session()
    
    # Define paths to your saved UNets for each fold
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
    
   


    print(f"  Normalizing data for fold {fold_idx + 1}...")
    

    print("  Skipping raw EEG normalization (replicating original method).")
    X_train_eeg_final_fold = X_train_raw_eeg_fold
    X_val_eeg_final_fold = X_val_raw_eeg_fold
    X_test_eeg_final_fold = X_test_raw_eeg_fold


    scaler_heatmap = StandardScaler()
    original_shape_heatmap = X_train_heatmap_fold.shape
    # Fit on flattened heatmap pixels of training data
    scaler_heatmap.fit(X_train_heatmap_fold.reshape(-1, 1))
    # Transform all sets
    X_train_heatmap_final_fold = scaler_heatmap.transform(X_train_heatmap_fold.reshape(-1, 1)).reshape(original_shape_heatmap)
    X_val_heatmap_final_fold = scaler_heatmap.transform(X_val_heatmap_fold.reshape(-1, 1)).reshape(X_val_heatmap_fold.shape)
    X_test_heatmap_final_fold = scaler_heatmap.transform(X_test_heatmap_fold.reshape(-1, 1)).reshape(X_test_heatmap_fold.shape)
    print("  Heatmap normalization complete.")

    # 5. Model Instantiation, Compile, Train
    tf.keras.backend.clear_session() 

    model = create_dual_branch_model(
        input_shape_raw_eeg,
        input_shape_heatmap,
        num_classes=1,
        eeg_transformer_embed_dim=64,
        eeg_transformer_num_heads=4,
        eeg_transformer_ff_dim=128,
        eeg_transformer_blocks=1,
        dropout_rate=0.3 # Match your original dropout
    )
    
    

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])

 
    fold_checkpoint_filepath = f'dual_branch_model_fold_{fold_idx+1}.weights.h5'
    fold_checkpoint_callback = ModelCheckpoint(
        filepath=fold_checkpoint_filepath, monitor='val_accuracy', save_best_only=True,
        save_weights_only=True, mode='max', verbose=1
    )
    rp_fold = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min', # Your patience=3
        min_delta=0.001, min_lr=1e-7
    )

    print(f"  Training model for fold {fold_idx + 1}...")
    history_fold = model.fit(
        [X_train_eeg_final_fold, X_train_heatmap_final_fold], # Pass inputs as list
        y_train_fold,
        epochs=30, # Max epochs
        batch_size=32,
        validation_data=([X_val_eeg_final_fold, X_val_heatmap_final_fold], y_val_fold), # Pass validation inputs as list
        callbacks=[fold_checkpoint_callback, rp_fold], # Use all 3 callbacks
        verbose=1 
    )

   
    print(f"Loading best weights from {fold_checkpoint_filepath} for evaluation...")
    model.load_weights(fold_checkpoint_filepath)

  
    eval_results = model.evaluate(
        [X_test_eeg_final_fold, X_test_heatmap_final_fold],
        y_test_fold,
        verbose=0
    )
    loss_fold, accuracy_fold = eval_results[0], eval_results[1]
    precision_fold = eval_results[2]
    recall_fold = eval_results[3]
    roc_auc_fold = eval_results[4]

    y_pred_probabilities = model.predict([X_test_eeg_final_fold, X_test_heatmap_final_fold], verbose=0).flatten()
    y_pred_classes = (y_pred_probabilities > 0.5).astype(int)

    # Store results
    fold_accuracies.append(accuracy_fold)
    fold_losses.append(loss_fold)
    fold_predictions_per_fold.append(y_pred_probabilities)
    fold_true_labels_per_fold.append(y_test_fold)
    fold_subject_ids_per_fold.append(test_subject_ids_for_epochs)

    # --- Subject-Level Metrics  ---
    subject_probs = defaultdict(list)
    subject_labels_map = {}
    
    for subj_id, prob, true_label in zip(test_subject_ids_for_epochs, y_pred_probabilities, y_test_fold):
        subject_probs[subj_id].append(prob)
        subject_labels_map[subj_id] = true_label
        
    y_true_subjects_fold, y_pred_subjects_fold, y_prob_subjects_fold = [], [], []
    for subj_id, probs in subject_probs.items():
        avg_prob = np.mean(probs)
        majority_vote = int(avg_prob > 0.5)
        y_pred_subjects_fold.append(majority_vote)
        y_true_subjects_fold.append(subject_labels_map[subj_id])
        y_prob_subjects_fold.append(avg_prob)
        
    y_true_subjects_fold = np.array(y_true_subjects_fold)
    y_pred_subjects_fold = np.array(y_pred_subjects_fold)
    
    print(f"\n--- Fold {fold_idx + 1} Subject-Level Metrics (Majority Vote) ---")
    acc_subj = accuracy_score(y_true_subjects_fold, y_pred_subjects_fold)
    prec_subj = precision_score(y_true_subjects_fold, y_pred_subjects_fold, zero_division=0)
    rec_subj = recall_score(y_true_subjects_fold, y_pred_subjects_fold, zero_division=0)
    f1_subj = f1_score(y_true_subjects_fold, y_pred_subjects_fold, zero_division=0)
    try:
        roc_auc_subj = roc_auc_score(y_true_subjects_fold, y_prob_subjects_fold)
    except ValueError:
        roc_auc_subj = np.nan
        
    print(f"Accuracy: {acc_subj:.4f}")
    print(f"Precision: {prec_subj:.4f}")
    print(f"Recall: {rec_subj:.4f}")
    print(f"F1-score: {f1_subj:.4f}")
    print(f"ROC-AUC: {roc_auc_subj:.4f}")
    
    subject_level_results.append({
        "accuracy": acc_subj, "precision": prec_subj, "recall": rec_subj,
        "f1": f1_subj, "roc_auc": roc_auc_subj
    })

  
    print(f"\n--- Fold {fold_idx + 1} Epoch-Level Metrics ---")
    print(f"  Test Loss: {loss_fold:.4f}")
    print(f"  Test Accuracy: {accuracy_fold:.4f}")
    print(f"  Test Precision: {precision_fold:.4f}")
    print(f"  Test Recall: {recall_fold:.4f}")
    print(f"  Test F1-score: {f1_score(y_test_fold, y_pred_classes, zero_division=0):.4f}")
    print(f"  Test ROC AUC: {roc_auc_fold:.4f}")

   
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history_fold.history['loss'], label='Training Loss'); plt.plot(history_fold.history['val_loss'], label='Validation Loss'); plt.title(f'Fold {fold_idx+1}: Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(history_fold.history['accuracy'], label='Training Accuracy'); plt.plot(history_fold.history['val_accuracy'], label='Validation Accuracy'); plt.title(f'Fold {fold_idx+1}: Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()


    cm = confusion_matrix(y_test_fold, y_pred_classes)
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1']); plt.title(f'Fold {fold_idx+1}: Epoch-Level CM'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()


print("\n--- 5-Fold Epoch-Level CV Summary (Dual Branch) ---")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

accs = [r["accuracy"] for r in subject_level_results]
precs = [r["precision"] for r in subject_level_results]
recs = [r["recall"] for r in subject_level_results]
f1s = [r["f1"] for r in subject_level_results]
rocs = [r["roc_auc"] for r in subject_level_results if not np.isnan(r["roc_auc"])]

print("\n--- 5-Fold Subject-Level CV Summary (Dual Branch - Majority Vote) ---")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"ROC-AUC: {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")


all_true_agg = []; all_pred_agg = []; all_prob_agg = []
fold_true_labels_flat = np.concatenate(fold_true_labels_per_fold)
fold_predictions_flat = np.concatenate(fold_predictions_per_fold)
fold_subject_ids_flat = np.concatenate(fold_subject_ids_per_fold)

subject_probs_agg = defaultdict(list)
subject_labels_map_agg = {}
for subj_id, prob, true_label in zip(fold_subject_ids_flat, fold_predictions_flat.flatten(), fold_true_labels_flat):
    subject_probs_agg[subj_id].append(prob)
    subject_labels_map_agg[subj_id] = true_label

for subj_id, probs in subject_probs_agg.items():
    avg_prob = np.mean(probs)
    majority_vote = int(avg_prob > 0.5)
    all_pred_agg.append(majority_vote)
    all_true_agg.append(subject_labels_map_agg[subj_id])
    all_prob_agg.append(avg_prob)

all_subject_true_agg = np.array(all_true_agg)
all_subject_pred_agg = np.array(all_pred_agg)
all_subject_prob_agg = np.array(all_prob_agg)

agg_acc_subj = accuracy_score(all_subject_true_agg, all_subject_pred_agg)
agg_prec_subj = precision_score(all_subject_true_agg, all_subject_pred_agg, zero_division=0)
agg_rec_subj = recall_score(all_subject_true_agg, all_subject_pred_agg, zero_division=0)
agg_f1_subj = f1_score(all_subject_true_agg, all_subject_pred_agg, zero_division=0)
agg_auc_subj = roc_auc_score(all_subject_true_agg, all_subject_prob_agg)
agg_cm_subj = confusion_matrix(all_subject_true_agg, all_subject_pred_agg)
if agg_cm_subj.size==4: 
    tn_agg, fp_agg, fn_agg, tp_agg = agg_cm_subj.ravel(); 
    agg_spec_subj = tn_agg / (tn_agg + fp_agg) if (tn_agg + fp_agg) > 0 else 0.0
else: 
    agg_spec_subj = np.nan

print(f"\n--- Aggregated SUBJECT-Level Metrics (Dual Branch - {len(all_subject_true_agg)} subjects total) ---")
print(f"Overall Subject Accuracy:    {agg_acc_subj:.4f}")
print(f"Overall Subject Precision:   {agg_prec_subj:.4f}")
print(f"Overall Subject Recall:      {agg_rec_subj:.4f}")
print(f"Overall Subject F1-Score:    {agg_f1_subj:.4f}")
print(f"Overall Subject ROC AUC:     {agg_auc_subj:.4f}")
print(f"Overall Subject Specificity: {agg_spec_subj:.4f}")

plt.figure(figsize=(6, 5)); 
sns.heatmap(agg_cm_subj, annot=True, fmt='d', cmap='Oranges', # Changed cmap
            xticklabels=['Predicted Control', 'Predicted ADHD'], 
            yticklabels=['Actual Control', 'Actual ADHD']); 
plt.title('Exp 2: Aggregated Subject CM (Dual Branch)'); plt.show()

print("\n--- Cross-Validation Finished ---")
