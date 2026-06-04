import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
TARGET_FOLD=4

eeg_data_path = 'add preprocessed eeg data path'

loaded_data_eeg = np.load(eeg_data_path)
all_subjects_X_eeg = loaded_data_eeg['X']
all_subjects_y = loaded_data_eeg['y']
subject_ids = loaded_data_eeg['subject_ids']


if all_subjects_X_eeg.ndim == 5 and all_subjects_X_eeg.shape[-1] == 1:
    all_subjects_X_eeg = np.squeeze(all_subjects_X_eeg, axis=-1)

if all_subjects_X_eeg.dtype != np.float32: 
    all_subjects_X_eeg = all_subjects_X_eeg.astype(np.float32)


input_shape_raw_eeg = all_subjects_X_eeg.shape[1:] 

# Helper Function for Subject Splitting
def get_data_for_subjects(data_array, labels_array, subject_ids_array, list_of_subject_ids_for_split):
    indices_for_split = np.isin(subject_ids_array, list_of_subject_ids_for_split)
    return data_array[indices_for_split], labels_array[indices_for_split], subject_ids_array[indices_for_split]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_unique_subject_ids = np.unique(subject_ids)

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(all_unique_subject_ids)):
    if fold_idx == TARGET_FOLD:
        current_test_subject_ids = all_unique_subject_ids[test_idx]
        train_val_subject_ids = all_unique_subject_ids[train_val_idx]
        break 

train_subject_ids, val_subject_ids = train_test_split(
    train_val_subject_ids, test_size=0.1, random_state=42
)


X_train_eeg, y_train, subj_train = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, train_subject_ids)
X_val_eeg, y_val, subj_val = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, val_subject_ids)
X_test_eeg, y_test, subj_test = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, current_test_subject_ids)

print(f"Train: {train_subject_ids} subjects {X_train_eeg.shape[0]} epochs | Val:{val_subject_ids}subjects {X_val_eeg.shape[0]} epochs | Test :{current_test_subject_ids} test subjects {X_test_eeg.shape[0]} epochs")


def build_finder_cnn(input_shape):
     return models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)), layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(32, activation='relu'),
        layers.Dropout(0.5), layers.Dense(1, activation='sigmoid') 
    ])

def build_simple_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1); c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2); c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    u4 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3); u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    u5 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4); u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    return models.Model(inputs, outputs)

def generate_dynamic_masks(model, X_data, batch_size=64):
    score_finder = BinaryScore(1)
    gradcam_finder = Gradcam(model, clone=True)
    
    last_conv_layer = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]
    
    y_mask = []
    for i in range(0, X_data.shape[0], batch_size):
        batch_epochs = X_data[i:i+batch_size]
        cam_finder = gradcam_finder(score_finder, batch_epochs, penultimate_layer=last_conv_layer, seek_penultimate_conv_layer=False)
        for single_cam in cam_finder:
            if single_cam.ndim > 2: single_cam = np.mean(single_cam, axis=-1)
            min_val, max_val = np.min(single_cam), np.max(single_cam)
            if max_val > min_val:
                threshold = min_val + 0.8 * (max_val - min_val) 
                mask = (single_cam >= threshold).astype(np.float32)
            else:
                mask = np.zeros_like(single_cam) 
            if np.sum(mask) == 0 and max_val > min_val:
                mask = (single_cam == max_val).astype(np.float32) 
            y_mask.append(np.expand_dims(mask, axis=-1))
    return np.array(y_mask)
def generate_dynamic_masks_score(model, X_data, batch_size=64):
    score_finder = BinaryScore(1)
    scorecam_finder = Scorecam(model, clone=True) 
    
   
    last_conv_layer = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]
    
    y_mask = []
    for i in range(0, X_data.shape[0], batch_size):
        batch_epochs = X_data[i:i+batch_size]
      
        cam_finder = scorecam_finder(score_finder, batch_epochs, penultimate_layer=last_conv_layer, seek_penultimate_conv_layer=False) 
        
        for single_cam in cam_finder:
            if single_cam.ndim > 2: single_cam = np.mean(single_cam, axis=-1)
            min_val, max_val = np.min(single_cam), np.max(single_cam)
            if max_val > min_val:
                threshold = min_val + 0.8 * (max_val - min_val) 
                mask = (single_cam >= threshold).astype(np.float32)
            else:
                mask = np.zeros_like(single_cam) 
            if np.sum(mask) == 0 and max_val > min_val:
                mask = (single_cam == max_val).astype(np.float32) 
            y_mask.append(np.expand_dims(mask, axis=-1))
            
    return np.array(y_mask)
def extract_unet_heatmaps(model, X_data, batch_size=64):
    score_unet = lambda output: output
    replace2linear = ReplaceToLinear()
    gradcam_unet = Gradcam(model, model_modifier=replace2linear, clone=False)
   
    last_conv_layer = [layer.name for layer in reversed(model.layers) if isinstance(layer, layers.Conv2D) and layer.filters > 1][0]
    
    heatmaps = []
    for i in range(0, X_data.shape[0], batch_size):
        batch_epochs = X_data[i:i+batch_size]
        cam_unet = gradcam_unet(score_unet, batch_epochs, penultimate_layer=last_conv_layer, seek_penultimate_conv_layer=False)
        for single_cam in cam_unet:
            if single_cam.ndim > 2: single_cam = np.mean(single_cam, axis=-1)
            min_val, max_val = np.min(single_cam), np.max(single_cam)
            if max_val > min_val: 
                single_cam = (single_cam - min_val) / (max_val - min_val)
            else: 
                single_cam = np.zeros_like(single_cam)
            heatmaps.append(np.expand_dims(single_cam, axis=-1))
    return np.array(heatmaps)
def extract_unet_heatmaps_score(model, X_data, batch_size=64):
    score_unet = lambda output: output
    replace2linear = ReplaceToLinear()
    scorecam_unet = Scorecam(model, model_modifier=replace2linear, clone=False) # <-- Changed to Scorecam
    
    
    last_conv_layer = [layer.name for layer in reversed(model.layers) if isinstance(layer, layers.Conv2D) and layer.filters > 1][0]
    
    heatmaps = []
    for i in range(0, X_data.shape[0], batch_size):
        batch_epochs = X_data[i:i+batch_size]
       
        cam_unet = scorecam_unet(score_unet, batch_epochs, penultimate_layer=last_conv_layer, seek_penultimate_conv_layer=False) 
        
        for single_cam in cam_unet:
            if single_cam.ndim > 2: single_cam = np.mean(single_cam, axis=-1)
            min_val, max_val = np.min(single_cam), np.max(single_cam)
            if max_val > min_val: 
                single_cam = (single_cam - min_val) / (max_val - min_val)
            else: 
                single_cam = np.zeros_like(single_cam)
            heatmaps.append(np.expand_dims(single_cam, axis=-1))
            
    return np.array(heatmaps)
def evaluate_hoyer_sparsity(heatmaps, name="Unknown Heatmaps"):
    """
    Evaluates the mathematical health and sparsity of CAMs using the Hoyer metric.
    Expects heatmaps shape: (N, 8, 8) or (N, 8, 8, 1)
    """
    if heatmaps.ndim == 4:
        heatmaps = np.squeeze(heatmaps, axis=-1)
        
    N_samples, H, W = heatmaps.shape
    N_elements = H * W
    sqrt_n = np.sqrt(N_elements)
    
   
    max_vals = np.max(heatmaps, axis=(1, 2))
    dead_masks = np.sum(max_vals < 1e-7)
    failure_rate = (dead_masks / N_samples) * 100
    
    
    alive_cams = heatmaps[max_vals >= 1e-7]
    hoyer_scores = []
    
    for cam in alive_cams:
        flat_cam = cam.flatten()
        flat_cam = flat_cam - np.min(flat_cam) # Ensure non-negative
        
        l1_norm = np.linalg.norm(flat_cam, ord=1)
        l2_norm = np.linalg.norm(flat_cam, ord=2)
        
        if l2_norm == 0:
            hoyer_scores.append(0.0)
            continue
            
        hoyer = (sqrt_n - (l1_norm / l2_norm)) / (sqrt_n - 1)
        hoyer_scores.append(hoyer)
        
    avg_hoyer = np.mean(hoyer_scores) if hoyer_scores else 0.0
    
    print(f"--- Evaluation: {name} ---")
    print(f"Dead Masks: {dead_masks}/{N_samples} ({failure_rate:.2f}% Failure)")
    print(f"Average Hoyer Sparsity: {avg_hoyer:.4f} (Closer to 1.0 is sharper/better)")
    print("-" * 40)
    
    return failure_rate, avg_hoyer

print("\n--- Loading Saved Models ---")
tf.keras.backend.clear_session()


finder_model_path = f'/finder_fold_{TARGET_FOLD}.keras'
unet_model_path = f'unet_fold_{TARGET_FOLD}.keras'

finder_model = tf.keras.models.load_model(finder_model_path)
unet_model = tf.keras.models.load_model(unet_model_path)

print(f" Loaded Finder and U-Net for Fold {TARGET_FOLD}")


test_loss, test_acc = finder_model.evaluate(X_test_eeg, y_test, verbose=0)
print(f" Finder Fold {TARGET_FOLD} Test Accuracy: {test_acc:.4f}")


print("\n--- Generating Finder Target Masks ---")


#y_mask_train = generate_dynamic_masks(finder_model, X_train_eeg) 
# y_mask_val = generate_dynamic_masks(finder_model, X_val_eeg) # Only needed if retraining
# y_mask_test = generate_dynamic_masks(finder_model, X_test_eeg)
#X_heatmap_train=generate_dynamic_masks(finder_model, X_train_eeg)
#X_heatmap_val=generate_dynamic_masks(finder_model, X_val_eeg)
#X_heatmap_test=generate_dynamic_masks(finder_model, X_test_eeg)

print("\n--- Extracting Final XaF Heatmaps ---")


X_heatmap_train = extract_unet_heatmaps(unet_model, X_train_eeg)
X_heatmap_val = extract_unet_heatmaps(unet_model, X_val_eeg)
X_heatmap_test = extract_unet_heatmaps(unet_model, X_test_eeg)


print("\n--- Evaluating Heatmap Sparsity (Hoyer Metric) ---")

_, hoyer_train = evaluate_hoyer_sparsity(X_heatmap_train, name="U-Net Train Heatmaps")
_, hoyer_val = evaluate_hoyer_sparsity(X_heatmap_val, name="U-Net Val Heatmaps")
_, hoyer_test = evaluate_hoyer_sparsity(X_heatmap_test, name="U-Net Test Heatmaps")



save_path = f'/kaggle/working/fold_{TARGET_FOLD}_xaf_data.npz'
np.savez(save_path, 
         X_heatmap_train=X_heatmap_train, X_eeg_train=X_train_eeg, y_train=y_train, subj_train=subj_train,
         X_heatmap_val=X_heatmap_val, X_eeg_val=X_val_eeg, y_val=y_val, subj_val=subj_val,
         X_heatmap_test=X_heatmap_test, X_eeg_test=X_test_eeg, y_test=y_test, subj_test=subj_test)

print(f"\n SUCCESSFULLY SAVED FOLD {TARGET_FOLD} DATA TO {save_path}")
