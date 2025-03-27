# Convolutional Neural Networks (CNN) for Fashion MNIST Classification

## Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify images from the **Fashion MNIST** dataset. The dataset contains **grayscale images** of **10 different clothing categories**, including dresses, shoes, bags, and more.

## Features
- Uses **Convolutional Layers** for automatic feature extraction.
- **Pooling Layers** for reducing spatial dimensions.
- **Dense (Fully Connected) Layers** for classification.
- **Visualization of model performance** with accuracy and loss plots.

## Prerequisites
Ensure you have the following dependencies installed:

- Python (>=3.7)
- TensorFlow / Keras
- NumPy
- Matplotlib

To install the required packages, run:


pip install numpy matplotlib tensorflow keras


## Dataset: Fashion MNIST
Fashion MNIST is a dataset of **70,000 grayscale images** (28x28 pixels) across **10 categories**:

1. **T-shirt/top**
2. **Trouser**
3. **Pullover**
4. **Dress**
5. **Coat**
6. **Sandal**
7. **Shirt**
8. **Sneaker**
9. **Bag**
10. **Ankle boot**

- **60,000 training images**
- **10,000 test images**

## Installation and Setup
1. Clone the repository:


   git clone 
   

2. Navigate to the project directory:

   cd Convolutional_Neural_Networks_.ipynb
 

3. Run the Jupyter Notebook or Python script:

   jupyter notebook


## Implementation Steps
1. **Load the Fashion MNIST dataset.**
2. **Normalize images** by scaling pixel values between 0 and 1.
3. **Build the CNN model** with convolutional, pooling, and dense layers.
4. **Compile the model** using categorical cross-entropy and Adam optimizer.
5. **Train the model** on the training dataset.
6. **Evaluate the model** on the test dataset.
7. **Visualize training accuracy and loss.**
8. **Make predictions** on sample test images.

## Example Code

# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Step 2: Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape to add a single color channel (grayscale)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize pixel values to the range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes in Fashion MNIST
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Step 7: Plot accuracy and loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Model Architecture
| Layer | Type | Filters | Activation | Output Shape |
|--------|------|---------|------------|--------------|
| Conv2D | Convolution | 32 | ReLU | (28, 28, 32) |
| MaxPool | Pooling | - | - | (14, 14, 32) |
| Conv2D | Convolution | 64 | ReLU | (14, 14, 64) |
| MaxPool | Pooling | - | - | (7, 7, 64) |
| Conv2D | Convolution | 128 | ReLU | (5, 5, 128) |
| Flatten | Flatten | - | - | (3200) |
| Dense | Fully Connected | 128 | ReLU | (128) |
| Dense | Output Layer | 10 | Softmax | (10) |

## Model Performance Metrics
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy
- **Visualization:** Training and validation accuracy/loss plots

## Visualizations
- **Accuracy Plot:** Monitors training & validation performance over epochs.
- **Loss Plot:** Helps analyze model convergence.

## Future Improvements
- Add **Batch Normalization** for improved training stability.
- Use **Data Augmentation** to improve model generalization.
- Experiment with **Dropout layers** to prevent overfitting.
- Implement **Transfer Learning** with pre-trained models (e.g., MobileNet, ResNet).

## License
This project is licensed under the MIT License.
