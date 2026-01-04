def build_simple_unet(input_shape):

    inputs = keras.Input(shape=input_shape)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1); c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2); c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    u4 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3); u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    u5 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4); u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = keras.Model(inputs, outputs); return model

"""Important Training Parameters:

unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


unet_historyc = unet_model.fit(X_all_epochs, y_mask_dynamic,
                                            epochs=20, 
                                            batch_size=32,
                                            validation_split=0.2, 
                                            verbose=1)

"""
