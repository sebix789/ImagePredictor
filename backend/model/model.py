import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import random
from dotenv import load_dotenv


# Build the model
def build_model():
    # Transer Learning Implementation
    base_model = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = tf.keras.models.Sequential([
        #Base model
        base_model,
        
        # Dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(120, activation='softmax')
    ])
    
    return model


# Preprocess Image
def preprocess_image(image, label, is_training=True):
    image = tf.image.resize(image, (224, 224))
    
    # Data Augmentation
    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
        image = tf.image.random_saturation(image, lower=0.95, upper=1.05)
        
        # Random Zoom Simulation
        zoom_factor = tf.random.uniform([], minval=0.9, maxval=1.1)
        new_size = tf.cast(tf.shape(image)[0:2], tf.float32) * zoom_factor
        image = tf.image.resize(image, tf.cast(new_size, tf.int32))
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        
        # Rotation
        rotations = random.choice([0, 1])
        image = tf.image.rot90(image, k=rotations)
        
    image = image / 255.0
    label = tf.one_hot(label, 120)
    return image, label


# Load dataset
def load_data():
    data, info = tfds.load('stanford_dogs', with_info=True, as_supervised=True)
    
    train_data = data['train'].map(lambda img, lbl: preprocess_image(img, lbl, is_training=True))
    test_data = data['test'].map(lambda img, lbl: preprocess_image(img, lbl, is_training=False))
    
    
    train_data = train_data.shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(64).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data


def fine_tunning(model, train_data, test_data):
    
    weights_file_path = os.getenv('KAGGLE_WEIGHTS_OUTPUT')
    
    base_model = model.layers[0]
    base_model.trainable = True
    
    for layer in base_model.layers[-50:]:
        layer.trainable = True
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file_path, save_best_only=True, monitor='val_loss', mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
    
    fine_tune_history = model.fit(
        train_data, epochs=50, 
        validation_data=test_data, 
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )
    
    return fine_tune_history

### Training configuration for Kaggle ###
def train_model(): 
    train_data, test_data = load_data()
    
    model_load_path = os.getenv('KAGGLE_MODEL_LOAD')
    weights_load_path = os.getenv('KAGGLE_WEIGHTS_LOAD')
    
    model_file_path = os.getenv('KAGGLE_MODEL_PATH')
    weights_file_path = os.getenv('KAGGLE_WEIGTHS_PATH')

    # Load or build the model
    if os.path.exists(model_load_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(weights_load_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Building new model...")
        model = build_model()

    # Load weights if they exist
    if os.path.exists(weights_file_path):
        print("Loading existing weights...")
        model.load_weights(weights_file_path)
    else:
        print("No existing weights found, starting with random initialization.")
        
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file_path, save_best_only=True, monitor='val_loss', mode='min')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5)

    # Train the model
    history = model.fit(
        train_data, epochs=10, 
        validation_data=test_data, 
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )
    
    print("Fine-Tunning applying...")
    fine_tune_history = fine_tunning(model, train_data, test_data)

    model.save(model_file_path)
    
    
    for key in fine_tune_history.history:
        history.history[key] += fine_tune_history.history[key]
        
        
    return model, history, test_data


def load_model():   
    model_path = os.getenv('MODEL_CURRENT')
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
    
    

#############################################################    
### If you plan to use TPU for trainings use this instead ###
# def load_data_with_TPU():
#     (train_data, test_data) = tfds.load(
#         'stanford_dogs',
#         split=['train', 'test'],
#         shuffle_files=True,
#         as_supervised=True,
#         with_info=True
#     )
    
#     train_data = train_data.map(lambda img, lbl: preprocess_image(img, lbl, is_training=True))
#     test_data = test_data.map(lambda img, lbl: preprocess_image(img, lbl, is_training=False))
    
    
#     train_data = train_data.batch(128)
#     test_data = test_data.batch(128)
    
#     return train_data, test_data
#############################################################
#############################################################
# def train_model_with_TPU():
#     # Initialize TPU strategy
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.TPUStrategy(tpu)
    
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
#             train_data, 
#             epochs=50, 
#             validation_data=test_data, 
#             callbacks=[early_stopping, model_checkpoint, lr_scheduler]
#         )

#     # Save the final model
#     model.save(model_file_path)
        
#     return model, history, test_data