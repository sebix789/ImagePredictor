import tensorflow as tf
import matplotlib.pyplot as plt
import os

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train, x_test = x_train /255.0, x_test / 255.0
    
    model = build_model()
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save('cifar_model.h5')
    
    return model, history, x_test, y_test

def load_model():   
    model_path = os.path.join(os.path.dirname(__file__), 'cifar_model.h5')
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
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    model, history, x_test, y_test = train_model()
    plot_history(history)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])    
    
    