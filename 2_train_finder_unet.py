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
"""The following code used to train fold wise finder and unet for XaF heatmap extraction"""

TARGET_FOLD = 2  #select the fold out of 5 fold cv to generate data


print(f"--- PREPARING XaF DATA FOR FOLD {TARGET_FOLD + 1}/5 ---")
print(f"{'='*50}\n")


# 2. LOAD RAW DATA

eeg_data_path = 'add path'


loaded_data_eeg = np.load(eeg_data_path)
all_subjects_X_eeg = loaded_data_eeg['X']
all_subjects_y = loaded_data_eeg['y']
subject_ids = loaded_data_eeg['subject_ids']

if all_subjects_X_eeg.ndim == 5 and all_subjects_X_eeg.shape[-1] == 1:
    all_subjects_X_eeg = np.squeeze(all_subjects_X_eeg, axis=-1)

if all_subjects_X_eeg.dtype != np.float32: 
    all_subjects_X_eeg = all_subjects_X_eeg.astype(np.float32)


input_shape_raw_eeg = all_subjects_X_eeg.shape[1:] 


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

=
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
    scorecam_finder = Scorecam(model, clone=True) # <-- Changed to Scorecam
    
    
    last_conv_layer = [layer.name for layer in model.layers if isinstance(layer, layers.Conv2D)][-1]
    
    y_mask = []
    for i in range(0, X_data.shape[0], batch_size):
        batch_epochs = X_data[i:i+batch_size]
        # <-- Changed call to scorecam_finder
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
    scorecam_unet = Scorecam(model, model_modifier=replace2linear, clone=False)
    
   
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

print("\n--- Training Finder Model ---")
tf.keras.backend.clear_session()
finder_model = build_finder_cnn(input_shape=input_shape_raw_eeg)
finder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


finder_history = finder_model.fit(
    X_train_eeg, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val_eeg, y_val), 
    verbose=1
)


final_train_acc = finder_history.history['accuracy'][-1]
final_val_acc = finder_history.history['val_accuracy'][-1]
test_loss, test_acc = finder_model.evaluate(X_test_eeg, y_test, verbose=0)
print(f" Finder Fold {TARGET_FOLD} Final Test Accuracy: {test_acc:.4f}")


print("\n--- Generating Dynamic Masks from Finder ---")
y_mask_train = generate_dynamic_masks(finder_model, X_train_eeg)
y_mask_val = generate_dynamic_masks(finder_model, X_val_eeg)


print("\n--- Training U-Net ---")
unet_model = build_simple_unet(input_shape=input_shape_raw_eeg)
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

unet_model.fit(X_train_eeg, y_mask_train, epochs=15, batch_size=32, validation_data=(X_val_eeg, y_mask_val), verbose=1)


print("\n--- Extracting Final XaF Heatmaps ---")
X_heatmap_train = extract_unet_heatmaps(unet_model, X_train_eeg)
X_heatmap_val = extract_unet_heatmaps(unet_model, X_val_eeg)

X_heatmap_test = extract_unet_heatmaps(unet_model, X_test_eeg)
import matplotlib.pyplot as plt


sample_idx = 0 

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


mask_img = axes[0].imshow(y_mask_train[sample_idx, :, :, 0], cmap='hot', vmin=0, vmax=1)
axes[0].set_title(f"Dynamic Mask (Finder)\nFold {TARGET_FOLD}, Sample {sample_idx}")
axes[0].axis('off')
fig.colorbar(mask_img, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Plot the Final XaF Heatmap (U-Net Output)
xaf_img = axes[1].imshow(X_heatmap_train[sample_idx, :, :, 0], cmap='hot', vmin=0, vmax=1)
axes[1].set_title(f"Final XaF Heatmap (U-Net)\nFold {TARGET_FOLD}, Sample {sample_idx}")
axes[1].axis('off')
fig.colorbar(xaf_img, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

save_path = f'/kaggle/working/fold_{TARGET_FOLD}_xaf_data.npz'
np.savez(save_path, 
         X_heatmap_train=X_heatmap_train, X_eeg_train=X_train_eeg, y_train=y_train, subj_train=subj_train,
         X_heatmap_val=X_heatmap_val, X_eeg_val=X_val_eeg, y_val=y_val, subj_val=subj_val,
         X_heatmap_test=X_heatmap_test, X_eeg_test=X_test_eeg, y_test=y_test, subj_test=subj_test)

print(f"\n SUCCESSFULLY SAVED FOLD {TARGET_FOLD} DATA TO {save_path}")



import os

save_dir = '/saved_models'
os.makedirs(save_dir, exist_ok=True)

finder_save_path = os.path.join(save_dir, f'finder_fold_{TARGET_FOLD}.keras')
unet_save_path = os.path.join(save_dir, f'unet_fold_{TARGET_FOLD}.keras')


finder_model.save(finder_save_path)
unet_model.save(unet_save_path)

print(f"\n SUCCESSFULLY SAVED MODELS FOR FOLD {TARGET_FOLD}")
print(f"Finder: {finder_save_path}")
print(f"U-Net: {unet_save_path}")
