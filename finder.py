def build_finder_cnn(input_shape):

     model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)), 
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)), 
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') 
    ])
     return model

"""Important training details:

checkpoint_filepath = 'best_model_checkpoint.weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,save_weights_only=True,
    monitor='val_accuracy',  
    mode='max',     
    save_best_only=True, 
    verbose=1            
)


finder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


finder_history = finder_model.fit(X_finder_train, y_finder_train,
                                epochs=20, 
                                batch_size=32,
                                validation_data=(X_finder_val, y_finder_val),callbacks=[model_checkpoint_callback],
                                verbose=1)
"""
