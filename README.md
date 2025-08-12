# MLP-FROM-SCRATCH

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/Framework-NumPy-orange)

A fully **from-scratch** implementation of a **Multi-Layer Perceptron (MLP)** in Python using only **NumPy**, complete with forward/backward propagation, multiple activation functions, optimizers, loss functions, and visualization support. This project is designed for learning, experimenting, and extending neural network fundamentals without relying on high-level frameworks like TensorFlow or PyTorch.



## 1. Introduction

**Purpose:**  
The goal of this project is to provide a clear, educational implementation of a Multi-Layer Perceptron (MLP) from scratch.  
It is suitable for:
- Students learning neural networks.
- Developers experimenting with custom architectures.
- Researchers prototyping lightweight models without heavy dependencies.

### Demo Screenshots

#### 1. Spiral Classification
![Spiral Classification](notebook/images/spiral_result.png)
> A 2D spiral dataset correctly classified by the trained MLP.

#### 2. MNIST Digit Recognition
![MNIST Results](notebook/images/mnist_predictions.png)
> Sample predictions on the MNIST dataset after training.

#### 3. Regression Task
![Regression Fit](notebook/images/regression_result.png)
> Fitting a non-linear function using the MLP regressor.



## 2. Key Features

- **Layer Implementations**:
  - Fully-connected (`Linear`) layers.
  - Multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax).
  - Dropout, Batch Normalization (1D and 2D).
- **Loss Functions**:
  - Mean Squared Error (MSE).
  - Cross Entropy Loss (with optional Softmax).
- **Optimizers**:
  - SGD, Momentum, RMSProp, Adagrad, Adam.
- **Training Utilities**:
  - Gradient clipping.
  - Mini-batch iterator.
  - Model save/load (Pickle-based).
- **Testing**:
  - Unit tests for activations, layers, and loss functions.



## 3. System Requirements

- **Python**: 3.8+
- **Required Libraries**:
  ```bash
  numpy>=1.21.0
  matplotlib>=3.4.0
  scikit-learn>=1.0.0
  tensorflow>=2.8.0
  

## 4. Installation & Usage
git clone https://github.com/honggquan24/MLP-from-scratch.git
cd MLP-FROM-SCRATCH

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

