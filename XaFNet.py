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

    inputs = Input(shape=input_shape_heatmap, name=f'{name_prefix}_input') 
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name=f'{name_prefix}_conv1')(inputs)
    x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool1')(x) 
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name=f'{name_prefix}_conv2')(x)
    x = MaxPooling2D((2, 2), name=f'{name_prefix}_pool2')(x) 
    x = Flatten(name=f'{name_prefix}_flatten')(x)
    # Output features (e.g., shape (None, 2*2*32 = 128))
    return Model(inputs=inputs, outputs=x, name=name_prefix)


def create_dual_branch_model(input_shape_raw_eeg_5d, input_shape_heatmap_3d, num_classes=1, #Note: this is the proposed XaF-Net
                             eeg_transformer_embed_dim=64,
                             eeg_transformer_num_heads=4,
                             eeg_transformer_ff_dim=128,
                             eeg_transformer_blocks=1,
                             dropout_rate=0.3): 


    input_eeg = Input(shape=input_shape_raw_eeg_5d, name='raw_eeg_input')
    input_heatmap = Input(shape=input_shape_heatmap_3d, name='heatmap_input')

    
    eeg_branch_output = cnn_lstm_transformer_block(
        input_eeg,
        name_prefix='eeg_branch',
        transformer_embed_dim=eeg_transformer_embed_dim,
        transformer_num_heads=eeg_transformer_num_heads,
        transformer_ff_dim=eeg_transformer_ff_dim,
        transformer_blocks=eeg_transformer_blocks,
        dropout_rate=0.3
    )


    heatmap_branch = build_heatmap_branch(input_shape_heatmap_3d)
    heatmap_branch_output = heatmap_branch(input_heatmap)

 
    merged_features = concatenate([eeg_branch_output, heatmap_branch_output], name='fused_features_after_branches')


    x = Dense(units=32, activation='relu', name='dense_fusion_1')(merged_features)
    x = Dropout(dropout_rate)(x) 
    x = Dense(units=16, activation='relu', name='dense_fusion_2')(x)
    x = Dropout(dropout_rate)(x)

    if num_classes == 1: 
        output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    else: 
        output_layer = Dense(units=num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=[input_eeg, input_heatmap], outputs=output_layer)
    return model

"""

Important training parameters:


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
        monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min', 
        min_delta=0.001, min_lr=1e-7
    )

    
    history_fold = model.fit(
        [X_train_eeg_final_fold, X_train_heatmap_final_fold], 
        y_train_fold,
        epochs=30,
        batch_size=32,
        validation_data=([X_val_eeg_final_fold, X_val_heatmap_final_fold], y_val_fold),
        callbacks=[fold_checkpoint_callback, rp_fold],
        verbose=1 
    )
"""


