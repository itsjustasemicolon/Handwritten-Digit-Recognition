# PyTorch Handwritten Digit Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.6-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. This project demonstrates the full machine learning workflow, including data loading, model training, evaluation, and GPU acceleration.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- MNIST dataset loading and preprocessing
- Custom CNN implementation
- GPU acceleration support
- Training and evaluation pipelines
- Model accuracy reporting

## Prerequisites
- Python 3.6+
- PyTorch 1.9+
- torchvision
- CUDA Toolkit (optional for GPU acceleration)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/pytorch-mnist.git
cd pytorch-mnist
```

2. Install dependencies:
```bash
pip install torch torchvision
```

## Dataset Overview
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels

```

## Model Architecture
The CNN model consists of:
- Two convolutional layers
- Max pooling layers
- Dropout regularization
- Fully connected layers

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

Key components:
- **Conv2d Layers**: Feature extraction
- **Max Pooling**: Dimensionality reduction
- **Dropout**: Regularization to prevent overfitting
- **Log Softmax**: Output layer activation

## Training & Evaluation

### Data Loading
```python
train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)
```

### Training Process
- Optimizer: Adam (learning rate 0.001)
- Loss Function: Cross Entropy Loss
- Epochs: 10
- Batch Size: 100

To start training:
```python
for epoch in range(1, 11):
    train(epoch)
    test()
```

### GPU Acceleration
The code automatically detects CUDA availability:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
```

## Results
After 10 epochs of training:
- Test loss: 0.042
- Accuracy: 98.7% (9870/10000 correct predictions)

Sample predictions:
```
[Insert example predictions with images]
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: The warning filter for `UserWarning` is included to suppress non-critical messages about internal dataset metadata. This can be safely ignored for experimental purposes but should be removed in production code.
