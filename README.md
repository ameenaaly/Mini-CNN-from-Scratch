# Convolutional Neural Network: Architecture, Implementation, and Results

## Objective
To design, implement, and evaluate a Convolutional Neural Network (CNN) for a classification task on the MNIST dataset. This project covers:

- Description of the CNN architecture.
- Implementation details using Python and NumPy.
- Training and testing results, including accuracy and loss.

---

## Dataset Preparation
### Loading and Preprocessing
The MNIST dataset is loaded using Keras. It contains 60,000 training and 10,000 testing images of handwritten digits (0-9). Each image has a size of 28x28 pixels.

Steps:
1. Normalize pixel values to the range [0, 1].
2. Convert labels into one-hot encoded vectors.

Example:
```python
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
num_classes = 10
y_one_hot_train = np.eye(num_classes)[y_train]
y_one_hot_test = np.eye(num_classes)[y_test]
```

---

## Network Architecture
The CNN consists of the following layers:

1. **Convolutional Layer 1**:
   - Kernel size: 3x3
   - Number of filters: 2
   - Activation: Sigmoid
   - Output: 26x26x2 feature maps

2. **Average Pooling Layers**:
   - Pool size: 2x2
   - Stride: 2
   - Output: 13x13x2 feature maps

3. **Flatten Layer**:
   - Input: Pooled feature maps
   - Output: Flattened vector of size 338

4. **1x1 Convolution Layer**:
   - Fully connected equivalent
   - Number of outputs: 10 (classes)

5. **Output Layer**:
   - Activation: Softmax
   - Output: Probabilities for each class

---

## Implementation Details

The CNN was implemented from scratch using Python and NumPy. Key components include:

### Convolution Operation
Performs dot products between the kernel and the input image to extract features.

```python
def convolve(image, kernel, stride=1):
    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = image[i * stride:i * stride + kernel.shape[0],
                          j * stride:j * stride + kernel.shape[1]]
            output[i, j] = np.sum(region * kernel)
    return output
```

### Average Pooling
Reduces the size of feature maps by averaging values within the pooling region.

```python
def average_pooling(featuremap, pool_size=2, stride=2):
    output_height = (featuremap.shape[0] - pool_size) // stride + 1
    output_width = (featuremap.shape[1] - pool_size) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = featuremap[i * stride:i * stride + pool_size,
                                j * stride:j * stride + pool_size]
            output[i, j] = np.mean(region)
    return output
```

### Forward Propagation
Passes input data through the network layers to generate predictions.

### Backpropagation
Updates weights by computing gradients of the loss with respect to each layer’s parameters.

### Training
Uses stochastic gradient descent (SGD) to minimize cross-entropy loss over 10 epochs.

---

## Results
The CNN was trained and tested on the MNIST dataset. The evaluation metrics include accuracy and average cross-entropy loss.

- **Learning Rate**: 0.001
- **Epochs**: 10
- **Test Results**:
  - Accuracy: **89%**
  - Average Loss: **0.3695**

Example output:
```text
Accuracy: 0.89
Average Loss: 0.3695
```

---

## Limitations
- The model's performance could be improved using techniques like batch normalization, dropout, or deeper architectures.
- Computational efficiency can be further optimized.

---

## Conclusion
This project demonstrated the design and implementation of a simple CNN for digit classification. Despite its simplicity, the model achieved competitive accuracy, showcasing the power of convolutional operations in feature extraction.

---

## References
- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)

