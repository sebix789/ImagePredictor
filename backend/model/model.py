import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os


def build_model(input_shape):
    
    # Build the model
    model = tf.keras.models.Sequential([
        # Model Input
        tf.keras.layers.Input(shape=input_shape),
        
        # CNN Layers
        tf.keras.layers.Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(256, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Conv2D(512, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        
        # Global Average Pooling instead of Flatten
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(120, activation='softmax')
    ])
    
    # SGD Optimizer instead of Nadam
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentun=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def preprocess_image(image, label, is_training=True):
    image = tf.image.resize(image, (224, 224))
    
    # Data Augmentation
    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random Zoom Simulation
        zoom_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
        new_size = tf.cast(tf.shape(image)[0:2], tf.float32) * zoom_factor
        image = tf.image.resize(image, tf.cast(new_size, tf.int32))
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        
        # Rotation
        random_angle = tf.random.uniform([], minval=-0.2, maxval=0.2)
        image = tfa.image.rotate(image, random_angle)
        
    image = image / 255.0
    label = tf.one_hot(label, 120)
    return image, label


def load_data():
    (train_data, test_data) = tfds.load(
        'stanford_dogs',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    train_data = train_data.map(lambda img, lbl: preprocess_image(img, lbl, is_training=True))
    test_data = test_data.map(lambda img, lbl: preprocess_image(img, lbl, is_training=False))
    
    
    train_data = train_data.batch(128).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(128).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data

### Training configuration for Kaggle ###
def train_model(): 
    # Load the data
    train_data, test_data = load_data()    
    
    # Path for Kaggle
    model_file_path = '/kaggle/working/classification_sequential_model.keras'

    # Load or build the model
    if os.path.exists(model_file_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_file_path)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentun=0.9)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Building new model...")
        model = build_model(224, 224, 3)
    
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_data, epochs=50, 
        validation_data=test_data, 
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )

    # Save the final model
    model.save(model_file_path)
    
    return model, history, test_data


def load_model():   
    model_path = os.path.join(os.path.dirname(__file__), 'classification_sequential_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = tf.keras.models.load_model(model_path)
    return model

def plot_history(history):
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    model, history, test_data = train_model()
    plot_history(history)
    
    score = model.evaluate(test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    
### If you plan to use TPU for trainings use this instead ###
# def train_model_with_TPU():
#     # Initialize TPU strategy
#     resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
#     tf.config.experimental_connect_to_cluster(resolver)
#     tf.tpu.experimental.initialize_tpu_system(resolver)
#     strategy = tf.distribute.TPUStrategy(resolver)
    
#     with strategy.scope():
#         # Load the data
#         train_data, test_data = load_data()
        
#         # Model file path for saving
#         model_file_path = '/kaggle/working/classification_sequential_model.keras'
        
#         # Load or build the model
#         if os.path.exists(model_file_path):
#             print("Loading existing model...")
#             model = tf.keras.models.load_model(model_file_path)
#         else:
#             print("Building new model...")
#             model = build_model((224, 224, 3))
        
#         # Callbacks
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#         model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
#         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)

#         # Train the model
#         history = model.fit(
#             train_data, epochs=50, 
#             validation_data=test_data, 
#             callbacks=[early_stopping, model_checkpoint, lr_scheduler]
#         )

#         # Save the final model
#         model.save(model_file_path)
        
#     return model, history, test_data