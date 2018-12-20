# Introduction to Pytorch

[TOC]

## 1. Welcome

<u>Flow</u>

1. tensors - the main data structure of PyTorch
   - how to crate tensors
   - how to do simple operations
   - how tensors interact with NumPy?
2. Autograd module - to calculate gradients for training neural networks
   - It is powerful as it does all the work of backpropagation by calculating the gradients at each operation in the network which you can then use to update the network weights
3. Use Pytorch to build a network and run data forward through it
4. Define loss and optimization method to train the neural network on a dataset of handwritten digits
5. How to test your network is able to generalize through validation?
6. How to use pre-trained networks to improve the performance of your classifier - in which the technique is called **transfer learning**

<u>Resources:</u>

All notebooks are available

```bash
git clone https://github.com/udacity/deep-learning-v2-pytorch.git
```

<u>Dependencies</u>

- PyTorhc v0.4 or newer, and torchvision

- Tutorial for installation

  - https://pytorch.org/get-started/locally/

- numpy and jupyter notebook

  ```bash
  conda install numpy jupyter notebook
  ```

- How to create environments and install packages
  https://conda.io/docs/
- GPU requirement (2 options)
  - A GPU
    - PyTorch uses a library called CUDA to accelerate the operations using the GPU
    - If we have a GPU that CUDA supports, we are able to install all the necessary libraries by installing PyTorch with conda

## 2. Single layer neural network

![single_layer_neural_network_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/single_layer_neural_network_1.jpg)

- How neural networks work?
  - input values x1, x2 multiply them by some weights w and bias
  - pass these input values h through this activation function gets you output y

<u>Mathematical representation:</u>

![single_layer_neural_network_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/single_layer_neural_network_2.jpg)

![single_layer_neural_network_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/single_layer_neural_network_3.jpg)

- The multiplication and sum is the same as a dot or inner product of two vectors, then we get our value h
- hass h through activation and gets our output y

![single_layer_neural_network_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/single_layer_neural_network_4.jpg)

- Vectors are an instance of a tensor, **it is just a generalization of vectors and matrices**
  - A tensor with only 1 dimension is a vector
    - single one-dimensional array of values
  - A tensor with 2 dimensions is a matrix
    - 2 directions from left to right and from top to the bottom
    - we can do operations
      - across the columns
      - along a row
      - across the rows
  - A tensor with 3 dimensional tensors
    - an image like an RGB color images (3D tensor)
      - there are some values for all the red and the green and the blue channels and so for every individual pixel, we have three values
  - 4D, 5D, 6D tensors and you name it

<u>How to import pyTorch?</u>

```python
import torch
```

<u>Create an activataion function sigmoid</u>

```python
def activation(x):
    return 1/(1+torch.exp(-x))
```

<u>Create Fake data</u>

```python
### Generate some data
torch.manual_seed(7) # Setting for an random number generation

# Features are 5 random normal variables, as samples from a normal distribution
features = torch.randn((1, 5)) # 2D tensor of 1 row and 5 columns
# True weights for our data, random normal variables again
weights = torch.randn_like(features) #it looks at the shape of this tensor and create it
# and a true bias term
bias = torch.randn((1, 1)) #1 row, 1 column = 1 value
```



## 3. Single layer neural networks solution

### Calculate the output of the network

```python
## Calculate the output of this network using the weights and bias tensors
y = activation(torch.sum(features*weights) + bias) # features times weight will be a element-wise multiplication

# alternative way
y = activation((features * weights).sum() + bias)
```

### matrix multiplication (more efficient)

```python
# more strict and simple, recommended
1. torch.mm() 
# it supports broadcasting, even put in tensors that have weird sizes, you could get an output that you are not expecting
2. torch.matmul() 
```

### Changing the shape of tensor - solve RuntimeError, size mismatch

error encountered: size mismatch --> the most popular error you will encounter when you are designing the architecture of neural networks

```python
RuntimeError: size mismatch, m1: [1 x 5], m2: [1 x 5] at /Users/soumith/minicondabuild3/conda-bld/pytorch_1524590658547/work/aten/src/TH/generic/THTensorMath.c:2033
```

Shape related operation

```python
# Check your tensor shape
tensor.shape

# 3 ways to reshape your tensor
# return a new tensor with teh same data as weights with size (a,b) sometimes
1. weights.reshape(a,b) 
# if the new shpae results in fewer elements than the original tensor, some elements will be removed from the tensor
2. weight.resize(a,b)
# return a new tensor with the same data as weights with size (a,b)
3. weights.view(a,b) #recommend
```

## 4. Networks Using Matrix Multiplication

![networks_matrix_multiplication_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/networks_matrix_multiplication_1.jpg)

- Stack up simple neural networks into multi-layer neural network can give network greater power to capture patterns and correlations in your data
- input = w1, w2, w3 --> vector x
- we have weights that connect our input to one hidden unit in this middle layer, hidden layer units
  - 2 units in the hidden layer
- We multiply our vector x by the first column to get the output h1
- We multiply our vector x by the second column to get the output h2

<u>Mathematical Representation</u>

![networks_matrix_multiplication_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/networks_matrix_multiplication_2.jpg)

The output for the above smal network is found by treating the hidden layer as inputs for the output unit

![networks_matrix_multiplication_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/networks_matrix_multiplication_3.jpg)

### general a network

```python
### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden) # how layers connect to each other
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output) # how layers connect to each other

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
```



## 5. Multilayer Networks Solution

### Calculate the output for multi-layer network using the W1 & W2 and biases B1 & B2

```python
h = activation(torch.mm(features, W1)+ B1)
y = activation(torch.mm(h, W2) + B2)
print(y)
```

### Convertion between Numpy arrays and Torch tensors

```python
# import numpy package
import numpy as np
# Create a numpy random array
a = np.random.rand(4,3)
# From numpy to torch tensors
b = torch.from_numpy(a)
# From torch tensors to numpy
b.numpy()
```

The memory is shared between the Numpy array and Torch tensor

```python
# Multiply PyTorch Tensor by 2, in place
b.mul_(2)
# Numpy array matches new values from Tensor
a
```

## 6. Neural Networks in PyTorch

<u>Dataset</u>: MNIST

![neural_networks_in_pytorch](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/neural_networks_in_pytorch.jpg)

- It is a whole bunch of grayscale handwritten digits (from 0 - 9)

  - It consists of images and their digit label

- Each of these image is 28 by 28 pixels

- Goal: identify what is the number is in these image

- This dataset is available through the **torchvision package**

  - procedure

    1. download and load the MNIST dataset

       ```python
       # Download and load the training data
       trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
       trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
       # 1. batch_size: number of images we get in one iteration from the data loader and pass through our network
       # 2. shuffle=True: tells it to shuffle the dataset every time we start going through the data loader again
       ```

       - it gives us back an object which I'm calling trainloader

         - we have the training data loaded into trainloader

         - and we make that an iterator with *iter(trainloader)*

           ```python
           dataiter = iter(trainloader)
           images, labels = dataiter.next() # This method returns the next input line
           print(type(images))
           print(images.shape)
           print(labels.shape)
           
           
           output:
           <class 'torch.Tensor'>
           torch.Size([64, 1, 28, 28])
           torch.Size([64])
           ```

    2. Later we will use this to loop through the dataset for training

       ```python
       for image, label in trainloader:
           ## do things with images and labels
       ```

<u>Shape of image</u>

- image is a tensor with size (64, 1, 28, 28)
  - 64 images
  - 1 color channels (grayscale)
  - 28 x 28  = shape of these image

<u>How do we build this network</u>

- 784 input units
  - it comes from the faact that with this type of network is called a **fully connected network**
    -  Each unit in one layer is connected to each unit in the next layer
  - we want to think of our inputs a just one vector
    - our image are actually 28 x 28 image
    - **how to change 28 x 28 to a vector into our network?**
      - We take this 28 x 28 image and flatten it into a vector then is's going to be 784 elements long
  - Take each of our batches, which is 64 x 1 x 28 x28 --> convert into a shape that is another tensor which shapes 64 x 784
    - flattening: 28 * 28, turning 2D images to 1D vector
- 256 hidden units
- 10 output units (one for each digit)
  - what we will do is calculate probabilities that the image is of any one digit or class
  - This ends up being **a discrete probability distribution over the classes (digits)** that tells us the most likely class for the image.

## 7. Neural Networks Solution

> **Exercise:** Flatten the batch of images `images`. Then build a multi-layer network with 784 input units, 256 hidden units, and 10 output units using random tensors for the weights and biases. For now, use a sigmoid activation for the hidden layer. Leave the output layer without an activation, we'll add one that gives us a probability distribution next.
>
> ```python
> # activation function
> def activation(x):
>     return 1/(1+torch.exp(-x))
> 
> # Flatten the input images
> inputs = images.view(images.shape[0],-1)
> 
> # Create parameters
> w1 = torch.randn(784,256)
> b1 = torch.randn(256)
> 
> w2 = torch.randn(256,10)
> b2 = torch.randn(10)
> 
> # hidden layer output
> h = activation(torch.mm(inputs, w1) + b1)
> # final output
> output = torch.mm(h,w2) + b2
> 
> print(output.shape)
> 
> >> output
> torch.Size([64, 10])
> ```
>
> <u>Remarks:</u>
>
> **Flatten the input image**
>
> - 1st element: batch size
>
> - 2nd element: keep the batch size, flatten the rest of the dimension (784 is also okay)
>
> - Put **-1** in the 2nd element can help to adjust the appropriate size to get the total no. of element



print(output)

![neural_network_solution_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/neural_network_solution_1.jpg)

- what we really want our network to tell is the probability of our different classes given some image
  - what is the most likely classes or digits that belong to this image
    ![neural_network_solution_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/neural_network_solution_2.jpg)
  - if it is a six, we want most of the probability goes to the bin of sixth class
  - In the image above, each of these different classes is roughly the same (uniform distribution)
  - So what is the way to convert 10 values calculated in our network and turn it into a proper probability distribution

## 8. Implementing Softmax Solution

- **Softmax function** (ask)

  ```python
  def softmax(x):
      return torch.exp(x)/torch.sum(torch.exp(x), dim = 1).view(-1, 1)
  	
      # explanation
      1. torch.exp(x) = 64 x 10
      2. torch.sum(torch.exp(x), dim = 1) = a vector of 64 elements
      # dim = 1 --> take the sum across columns
      # division means it is going to try to divide every element in this 	tensor by all 64 of these values, it's going to give us a 64 x 64   	tensor
      # want to reshape the tensor to have 64 rows
  
  ```

### Construction of neural network using nn module in pyTorch

```python
from torch import nn
 
class Network(nn.Module): # inheriting from nn.Module
    def __init__(self):
        super().__int__() # create a class that tracks the architecture and provide a lot of useful methods and attributes
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units- one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
       # Pass the input tensor through each of our operations
       x = self.hidden(x)
       x = self.sigmoid(x)
       x = self.ouput(x)
       x = self.softmax(x)
        
       return x
```

> Reminder:
>
> ```python
> self.hidden = nn.Linear(784, 256)
> ```
>
> This line creates a module for a linear transformation, xW+bxW+b, with 784 inputs and 256 outputs and assigns it to `self.hidden`. The module **automatically creates the weight and bias tensors which we'll use in the `forward` method**. You can access the weight and bias tensors once the network (`net`) is created with `net.hidden.weight` and `net.hidden.bias`.
>
> ```python
> self.sigmoid = nn.Sigmoid()
> self.softmax = nn.Softmax(dim=1)
> ```
>
> These are operations for the sigmoid activation and softmax output. Setting `dim=1` in `nn.Softmax(dim=1)` calculates softmax across the columns.
>
> ```python
> x = self.hidden(x)
> x = self.sigmoid(x)
> x = self.output(x)
> x = self.softmax(x)
> ```
>
> - input tensor `x` is passed through each operation a reassigned to `x`
> - nput tensor goes through the hidden layer, then a sigmoid function, then the output layer, and finally the softmax function.
> - It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build.
> -  The order in which you define things in the `__init__` method doesn't matter, but you'll need to **sequence** the operations correctly in the `forward` method.

Check the Network Architecture

```python
# Create the network and look at it's text representation
model = Network()
model

>> output
Network(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
  (sigmoid): Sigmoid()
  (softmax): Softmax()
)
```



### construct network by using torch.nn.functional module

```python
import torch.nn.fucntional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
```

### Different Activation Functions

![implementing_softmax_solution_1](/Users/anson/Desktop/Courses/Year%202_FirstSem/Pytorch%20Challenge/image/implementing_softmax_solution_1.jpg)

- These activation functions should typicall non-linear, so the network is able to learn non-linear correlations and patterns
- The most commmonly used is the ReLU, the simplest non-linear function
- Networks tend to train a lot faster when we use ReLU

## 9. Network Architectures in PyTorch

### Training Neural Networks

![network_architecture_in_pytorch_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/network_architecture_in_pytorch_2.jpg)

- we are going to treat our neural network as a universal function approximator
- we have desired input and desired output of this function
  - you have the correct dataset of these imags that are labeled with the correct ones
- **eventually our neural network will build to approximate this function that is converting these imaages into this probability distribution here**
- **we can build a neural network and then we are going to <u>give it the inputs and outputs and adjust the weights of that network</u> so that it approximates this function**

### What do we need for training neural network

#### 1. loss function

-  A measure of our prediction error. For example, the mean squared loss is often used in regression and binary classification problems

  ![network_architecture_in_pytorch_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/network_architecture_in_pytorch_1.jpg)

- We can adjust our weights such that this loss is minimized

- Our whole goal is to adjust our network parameters to minimize our loss

#### 2. Gradient Descent

![network_architectures_pytorch_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/network_architectures_pytorch_3.jpg)

- We can find configurations where the loss is at a minimum and the network is able to predict the correct labels with high accuracy, and the process is called gradient descent
- **The gradient is the slope of the loss function and points in the direction of fastest change

#### 4. Backpropagation doing gradient descent in multi-layer network

![network_architecture_in_pytorch_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/network_architecture_in_pytorch_4.jpg)

- It is an application of the chain rule from calculus
- It goes through this forward pass through the network to calculate our loss
- some feature input x and then it goes through this linear transformation which depends on our weights and biases and then that goes in
- So if we make a small change in our weights here, w1, it's going to propagate through the network and end up like results in a small change in our loss
  - backpropagation goes in the opposite direction
  - each step we multiply the incoming gradient with the gradient of the operation itself
- In the backpropagation side
  - this is the gradient of the loss with respect to the second linear transformation
  - then we pass that backwards again and if we multiply it by the loss of this L2
    - this is the linear transformation with respect to the outputs of our activation function
  - If you multiply this gradient by the gradient coming from the loss
    - this gradient can be passed back to this softmax function
  - we pass it backwards to the previous operation
  - overall:
    - take the gradient of the loss function
    - pass it back to the previous operation , mutliply the gradient there
    - pass the total gradient to the softmax, so on and so forth
    - eventually we we back propagate to our weight
      - **calculate the gradient of our weigh, gradient point to the direction with the fastest change to maximize our loss**
      - **so we will substract the gradient off from our weight**
        ![network_architectures_in_pytorch_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/network_architectures_in_pytorch_5.jpg)
        - substract with a multiple of learning rate, so we are not taking too large step
    - Finally it will give out a new set of weights, result in a smaller loss
      - because if the each step is too large, the loss will kind of bound around the minimum and never settles in the minimum point
  - Sequence (until we minimize our loss
    - forward --> loss --> backpropagate --> weight - gradient --> forward --> loss --> backpropagate --> weight - gradient

### Losses in PyTorch 

- Through the `nn` module, PyTorch provides losses such as the cross-entropy loss

  ```python
  # Loss for classification problem
  criterion = nn.CrossEntropyLoss()
  ```

><u>Remarks</u>
>
>The criterion combines ``nn.LogSoftmax()`` and ``nn.NLLLoss()``in one single class
>
>The input is expected to contain scores for each class

*NOTE: 

- **We need to pass in the raw output of our network into the loss, not the output of the softmax function**
- The raw output is usually called the logits or scores of each class
  - We use the logits because softmax gives you probabilities which will often be very close to zero or one but floating-point numbers can't accurately represent values near zero or one
  - It's usually best to avoid doing calculations with probabilities, typically we use log-probabilities

```python
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

```python
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)

>> output
tensor(2.2810)
```

## 10. Network Architectures Solution

- According to experience, it's more convenient to build the model with a log-softmax output using `nn.LogSoftmax` or `F.log_softmax`
  - Then you can get the actual probabilites by taking the exponential `torch.exp(output)`. 
- **With a log-softmax output, you want to use the negative log likelihood loss, `nn.NLLLoss`**

```python
# TODO: Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)) 

# TODO: Define the loss
criterion = nn.NLLLoss()

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
```

### Autograd_PyTorch

- It automatically calculate the gradients of tensor
- we can use it to calculate the gradients of all our parameters with respect to the loss
- **Autograd works by keeping track of operations performed on tensors, then going backwards through those operations, calculating gradients along the way

Functions

1. To make sure PyTorch keeps track of operations on a tensor and calculates the gradients

   ```python
   requires_grad = True
   
   x.requires_grad_(True)
   ```

2. Turn off gradients for a block of code with the ``torch.no_grad()`` content:

   ```Python
   x = torch.zeros(1, requires_grad = True)
   >>> with torch.no_grad():
   		y = x * 2
   >>> y.requires_grad
   False
   ```

3. Turn on or off gradients altogether

   ```python
   torch.set_grad_enabled(True|False)
   ```


Example 1

```python
# Create a random tensor x
x = torch.randn(2,2, requires_grad=True) # requires_grad = True starts from here
print(x)

output:
tensor([[-0.4381, -0.0605],
        [-0.0246, -1.3241]], requires_grad=True)

# perform square operation to x
y = x**2
print(y)

output:
tensor([[0.1920, 0.0037],
        [0.0006, 1.7533]], grad_fn=<PowBackward0>)

#  see the operation that created y, a power operation PowBackward0
print(y.grad_fn)

output:
<PowBackward0 object at 0x10e2744e0>

# reduce the tensor y to a scalar value, the mean
z = y.mean()
print(z)

output:
tensor(0.4874, grad_fn=<MeanBackward1>)

#  check the gradients for x and y but they are empty currently.
print(x.grad)

output:
None

# Calculate the gradient for z with respect to x
z.backward()
print(x.grad)
print(x/2)

oupput:
tensor([[-0.2191, -0.0303],
        [-0.0123, -0.6621]])
tensor([[-0.2191, -0.0303],
        [-0.0123, -0.6621]], grad_fn=<DivBackward0>)
```

- For training, we need the gradients of the weights with respect to the cost
- With PyTorch, we run data forward through the network to calculate the loss, then, go backwards to calculate the gradients with respect to the loss.

### Loss and Autograd together

- we can only perform back propagation after we have passed our data to get the loss from the loss function

```python
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)

# Before
print('Before backward pass: \n', model[0].weight.grad) # model[0] = 1st Linear Operation
# Backward Propagation
loss.backward()
# After
print('After backward pass: \n', model[0].weight.grad)

output:
Before backward pass: 
 None
After backward pass: 
 tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0006, -0.0006, -0.0006,  ..., -0.0006, -0.0006, -0.0006],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0001,  0.0001,  0.0001,  ...,  0.0001,  0.0001,  0.0001],
        [-0.0027, -0.0027, -0.0027,  ..., -0.0027, -0.0027, -0.0027],
        [-0.0029, -0.0029, -0.0029,  ..., -0.0029, -0.0029, -0.0029]])   
```

### Optimizer: update the weight with the gradients

- Define an optimizer

```python
from torch import optim

# Optimzers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

- Entire learning step before looping through all the data
  1. Make a forward pass through the network
  2. Use the network output to calculate the loss
  3. Perform a backward pass through the network with ``loss.backward()`` to calculate gradients
  4. Take a step with the optimizer to update the weights
- Single step Learnnig example

```python
print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated *****
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)
```

**REMARKS**: When you do multiple backwards passes with the same parameters, the gradients are accumulated. This means that you need to zero the gradients on each training pass or you'll retain gradients from previous training batches.

```python
# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)
```

## 11. Training a Network Solution

### Training for Real

```python

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass****
        # clean up the gradient
        optimizer.zero_grad()
        # pass the image to the model (feed forward)
        output = model.forward(images)
        # loss
        loss = criterion(output, labels)
        # backpropagation
        loss.backward()
        # update the weight by stochastic gradient descent
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

output:
Training loss: 1.8813433436188363
Training loss: 0.8589788045265527
Training loss: 0.5318579362399543
Training loss: 0.4300458642211296
Training loss: 0.3857253724451004
```

#### Apply the trained model to a new image

```python
%matplotlib inline
import helper

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)
```

Output:

![train_a_network_solution_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/train_a_network_solution_1.jpg)

## 12. Fashion-MINIST

- **Building the network**

  ```python
  # TODO: Define your network architecture here
  class Classifier(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(784, 256)
          self.fc2 = nn.Linear(256, 128)
          self.fc3 = nn.Linear(128, 64)
          self.fc4 = nn.Linear(64, 10)
          
      def forward(self, x):
          # make sure input tensor is flattened
          x = x.view(x.shape[0], -1)
          
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = F.relu(self.fc3(x))
          x = F.log_softmax(self.fc4(x), dim=1)
          
          return x
  ```

- **Train the network**

- Then write the training code. Remember the training pass is a fairly straightforward process:

  - Make a forward pass through the network to get the logits
  - Use the logits to calculate the loss
  - Perform a backward pass through the network with `loss.backward()` to calculate the gradients
  - Take a step with the optimizer to update the weights

- ```python
  # TODO: Create the network, define the criterion and optimizer
  model = Classifier()
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.003)
  
  # TODO: Train the network here
  epochs = 5
  
  for e in range(epochs):
      running_loss = 0
      for images, labels in trainloader:
          log_ps = model(images)
          loss = criterion(log_ps, labels)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item()
      else:
          print(f"Training loss: {running_loss/len(trainloader)}")
  ```

## 13. Inference and Validation

- **Inference**

  - You have a trained network, you can use it for making predictions

- **Overfitting**

  - Neural networks have a tendency to perform *<u>too well</u>* on the training data and <u>aren't able to generalize to data that hasn't been seen before</u>.

- **How to solve overfitting?**

  - we measure the performance on data not in the training set called the **validation** set. 
  - Ways
    - Through regularization e.g. dropout
    - monitoring the validation performance during training

- **Loading the data through tochvision**

  ```python
  # Download and load the training data
  trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
  
  # Download and load the test data
  testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
  ```

- **Goal of validation**

  - measure the model's performance on data that isn't part of the training set

  - Performance here is up to the developer to define 

    - **Accuracy**: the percentage of classes the network predicted correctly
    - **Precision**
    - **Recall**
    - **Top-5 error rate**

    ```python
    model = Classifier()
    
    images, labels = next(iter(testloader))
    # Get the class probabilities
    ps = torch.exp(model(images))
    # Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
    print(ps.shape)
    ```

    With the probabilities, we can get the most likely class using the `ps.topk` method. 

     - returns the kk highest values

     - since we just want the most likely classes

        - use ps.topk(1)

          ```python
          top_p, top_class = ps.topk(1, dim=1)
          # Look at the most likely classes for the first 10 examples
          print(top_class[:10,:])
          
          >>ouput
          tensor([[3],
                  [5],
                  [5],
                  [5],
                  [3],
                  [5],
                  [5],
                  [5],
                  [5],
                  [5]])
          ```

  - **Check if the predicted classes match the labels**

    - This is simple to do by equating `top_class` and `labels`

    - `top_class` and `labels` must have the same shape

      ```python
      equals = top_class == labels.view(*top_class.shape)
      
      # equals will have shape (64, 64)
      ```

  - **Calculate the percentage of correct predicitons**

    ```python
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')
    
    # Note
    # 1)  equals has binary values, either 0 or 1. This means that if we just sum up all the values and divide by the number of values, we get the percentage of correct predictions.
    # 2) This happens because equals has type torch.ByteTensor but torch.mean isn't implement for tensors with that type.
    # 3) we'll need to convert equals to a float tensor
    
    ```

  - **Train our network and include our validation pass**

    -  Since we're not updating our parameters in the validation pass, we can speed up the by turning off gradients using `torch.no_grad()`:

      ```python
       turn off gradients
      with torch.no_grad():
          # validation pass here
          for images, labels in testloader:
              ...
      ```

    - Validation loop full example

      ```python
      model = Classifier()
      criterion = nn.NLLLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.003)
      
      epochs = 30
      steps = 0
      
      train_losses, test_losses = [], []
      for e in range(epochs):
          running_loss = 0
          for images, labels in trainloader:
              
              optimizer.zero_grad()
              
              log_ps = model(images)
              loss = criterion(log_ps, labels)
              loss.backward()
              optimizer.step()
              
              running_loss += loss.item()
              
          else:
              test_loss = 0
              accuracy = 0
              
              # Turn off gradients for validation, saves memory and computations
              with torch.no_grad():
                  for images, labels in testloader:
                      log_ps = model(images)
                      test_loss += criterion(log_ps, labels)
                      
                      ps = torch.exp(log_ps)
                      top_p, top_class = ps.topk(1, dim=1)
                      equals = top_class == labels.view(*top_class.shape)
                      accuracy += torch.mean(equals.type(torch.FloatTensor))
                      
              train_losses.append(running_loss/len(trainloader))
              test_losses.append(test_loss/len(testloader))
      
              print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
      ```

    - Plot the training loss and validation loss

      ```python
      %matplotlib inline
      %config InlineBackend.figure_format = 'retina'
      
      import matplotlib.pyplot as plt
      
      plt.plot(train_losses, label='Training loss')
      plt.plot(test_losses, label='Validation loss')
      plt.legend(frameon=False)
      ```

## 14. Dropout Solution

Overfitting**

- ![inference_and_validation_1](/Users/anson/Desktop/Courses/Year%202_FirstSem/Pytorch%20Challenge/image/inference_and_validation_1.jpg)

- The network learns the training set better and better, resulting in lower training losses.

- It starts having problems generalizing to data outside the training set leading to the validation loss increasing.

-  **The ultimate goal of any deep learning model is to make predictions on new data, so we should strive to get the lowest validation loss possible.**

- Solutions

  - Early stopping

    -  One option is to use the version of the model with the lowest validation loss, here the one around 8-10 training epochs.

  - **Dropout**

    - we randomly drop input units. 

    - This forces the network to share information between weights, increasing it's ability to generalize to new data.

      ```python
      class Classifier(nn.Module):
          def __init__(self):
              super().__init__()
              self.fc1 = nn.Linear(784, 256)
              self.fc2 = nn.Linear(256, 128)
              self.fc3 = nn.Linear(128, 64)
              self.fc4 = nn.Linear(64, 10)
      
              # Dropout module with 0.2 drop probability
              self.dropout = nn.Dropout(p=0.2)
      
          def forward(self, x):
              # make sure input tensor is flattened
              x = x.view(x.shape[0], -1)
      
              # Now with dropout
              x = self.dropout(F.relu(self.fc1(x)))
              x = self.dropout(F.relu(self.fc2(x)))
              x = self.dropout(F.relu(self.fc3(x)))
      
              # output so no dropout here
              x = F.log_softmax(self.fc4(x), dim=1)
      
              return x
      ```

    - **Training mode and evaluation mode**

      - During training we want to use dropout to prevent overfitting, but during inference we want to use the entire network

        - **we need to turn off dropout during validation, testing, and whenever we're using the network to make predictions**

        - To do this, you use `model.eval()`. This sets the model to evaluation mode where the dropout probability is 0.

          ```python
          # set model to evaluation mode
              model.eval()
          ```

        - You can turn dropout back on by setting the model to train mode with `model.train()`.

          ```python
          # set model back to train mode
          model.train()
          ```

        - General pattern

          - he pattern for the validation loop will look like this, where you turn off gradients, set the model to evaluation mode, calculate the validation loss and metric, then set the model back to train mode.

            ```python
            # turn off gradients
            with torch.no_grad():
            
                # set model to evaluation mode
                model.eval()
            
                # validation pass here
                for images, labels in testloader:
                    ...
            
            # set model back to train mode
            model.train()
            ```

          - Full example

            ```python
            model = Classifier()
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.003)
            
            epochs = 30
            steps = 0
            
            train_losses, test_losses = [], []
            for e in range(epochs):
                running_loss = 0
                for images, labels in trainloader:
                    
                    optimizer.zero_grad()
                    
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                else:
                    test_loss = 0
                    accuracy = 0
                    
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        model.eval()
                        for images, labels in testloader:
                            log_ps = model(images)
                            test_loss += criterion(log_ps, labels)
                            
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                    model.train()
                    
                    train_losses.append(running_loss/len(trainloader))
                    test_losses.append(test_loss/len(testloader))
            
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                          "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            ```

      - Inference

        - now we need to remember to set the model in inference mode with `model.eval()`. You'll also want to turn off autograd with the `torch.no_grad()` context.

          ```python
          # Import helper module (should be in the repo)
          import helper
          
          # Test out your network!
          
          model.eval()
          
          dataiter = iter(testloader)
          images, labels = dataiter.next()
          img = images[0]
          # Convert 2D image to 1D vector
          img = img.view(1, 784)
          
          # Calculate the class probabilities (softmax) for img
          with torch.no_grad():
              output = model.forward(img)
          
          ps = torch.exp(output)
          
          # Plot the image and probabilities
          helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
          ```

## 15. Saving and Loading Models

This is importatn because you will often load previous trained models to use in making predictions or to continue training on new data

<u>Scenario</u>

1. Train a network

   - Moved the model architecture and training code to a file called `fc_model.train`

     - we can easily create a fully-connected network with `fc_model.Network`

       ```python
       # Create the network, define the criterion and optimizer
       
       # parameters: inputs, outputs, hidden layers
       model = fc_model.Network(784, 10, [512, 256, 128])
       criterion = nn.NLLLoss()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       ```

     - train the network using `fc_model.train`

   - ```python
     fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)
     ```

2. Saving and loading networks

   - **it's impractical to train a network every time you need to use it.**

   - <u>Instead, we can save trained networks then load them later to train more or use them for predictions.</u>

   - The parameters for PyTorch networks are stored in a model's `state_dict`

     - We can see the state dict contains the weight and bias matrices for each of our layers.

       ```python
       print("Our model: \n\n", model, '\n')
       print("The state dict keys: \n\n", model.state_dict().keys())
       
       >> output:
        
       Our model: 
       
        Network(
         (hidden_layers): ModuleList(
           (0): Linear(in_features=784, out_features=512, bias=True)
           (1): Linear(in_features=512, out_features=256, bias=True)
           (2): Linear(in_features=256, out_features=128, bias=True)
         )
         (output): Linear(in_features=128, out_features=10, bias=True)
         (dropout): Dropout(p=0.5)
       ) 
       
       The state dict keys: 
       
        odict_keys(['hidden_layers.0.weight', 'hidden_layers.0.bias', 'hidden_layers.1.weight', 'hidden_layers.1.bias', 'hidden_layers.2.weight', 'hidden_layers.2.bias', 'output.weight', 'output.bias'])
       ```

   - The simplest thing to do is simply save the state dict with `torch.save`

     ```python
     # save it to a file 'checkpoint.pth'
     torch.save(model.state_dict(), 'checkpoint.pth')
     ```

   - Load the state dict in to the network, you do `model.load_state_dict(state_dict)`.

     ```python
     model.load_state_dict(state_dict)
     ```

   - However, loading the state dict works only if the model architecture is exactly the same as the checkpoint architecture, it fails if we create a model with a different architecture

     ```python
     # Failed example
     
     # Try this, in which the saved one's architecture is (784, 10, [512, 256, 128])
     model = fc_model.Network(784, 10, [400, 200, 100])
     # This will throw an error because the tensor sizes are wrong!
     model.load_state_dict(state_dict)
     ```

   - **Therefore, we need to rebuild the model exactly as it was when trained.**

   - Information about the model architecture needs to be saved in the checkpoint, along with the state dict.

   - To do this, we <u>build a dictionary</u> with all the information you need to compeletely rebuild the model

     ```python
     # the hidden_layers part is a little bit more tricky
     checkpoint = {'input_size': 784,
                   'output_size': 10,
                   'hidden_layers': [each.out_features for each in model.hidden_layers],
                   'state_dict': model.state_dict()}
     
     torch.save(checkpoint, 'checkpoint.pth')
     ```

   - Write a function to load checkpoints

     ```python
     def load_checkpoint(filepath):
         checkpoint = torch.load(filepath)
         model = fc_model.Network(checkpoint['input_size'],
                                  checkpoint['output_size'],
                                  checkpoint['hidden_layers'])
         model.load_state_dict(checkpoint['state_dict'])
         
         return model
     
     # load the function
     model = load_checkpoint('checkpoint.pth')
     print(model)
     
     >> output:
         Network(
       (hidden_layers): ModuleList(
         (0): Linear(in_features=784, out_features=400, bias=True)
         (1): Linear(in_features=400, out_features=200, bias=True)
         (2): Linear(in_features=200, out_features=100, bias=True)
       )
       (output): Linear(in_features=100, out_features=10, bias=True)
       (dropout): Dropout(p=0.5)
     )
     ```


## 16. Loading Image Data

How to deal with full-sized images (like the one you would get from your iphone)**

- Load images

  - **The easiest way to load image data is with `datasets.ImageFolder` from `torchvision**`

    ```python
    dataset = datasets.ImageFolder('path/to/data', transform=transform)
    ```

    -  `'path/to/data'` is the file path to the data directory

    - Transform is a sequence of processing steps built with the transforms module from torch vision

    - `ImageFolder` expects the files and directories to be constructed like

      ```python
      # Each class has it's own directory for the images
      # The images are labeled with the class taken from the directory name
      root/dog/xxx.png
      root/dog/xxy.png
      root/dog/xxz.png
      
      # image 123.png would be loaded with the class label cat
      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png
      ```

  - **Transform**

    - When you load in the data with `ImageFolder`, you'll need to define some transforms
    - Images are different sizes but we'll need them to all be the same size for training
      - You can either resize them with `transforms.Resize()` or crop with `transforms.CenterCrop()`, `transforms.RandomResizedCrop()`, etc.
    - We'll also need to convert the images to PyTorch tensors with `transforms.ToTensor()`. 
    - **Typically you'll combine these transforms into a pipeline with `transforms.Compose()`**
      - it  accepts a list of transforms and runs them in sequence. It looks something like this to scale, then crop, then convert to a tensor:
      - documentation for transform
        https://pytorch.org/docs/stable/torchvision/transforms.html

  - **Data Loader**

    - With the `ImageFolder` loaded, you have to pass it to a [`DataLoader`](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).

      - The `DataLoader` takes a dataset (such as you would get from `ImageFolder`) and returns batches of images and the corresponding labels.

        - You can set various parameters like the batch size and if the data is shuffled after each epoch.

          ```python
          dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
          ```

        - **Here `dataloader` is a [generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). To get data out of it, you need to loop through it or convert it to an iterator and call `next()`.**

          ```python
          # Looping through it, get a batch on each loop 
          for images, labels in dataloader:
              pass
          
          # Get one batch
          images, labels = next(iter(dataloader))
          ```

        - Example

          ```python
          data_dir = 'Cat_Dog_data/train'
          
          # resize = turn size to 255x255
          # CenterCrop = crop in the cender to 224x224
          transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
          dataset = datasets.ImageFolder(data_dir, transform=transform)
          # shuffle = True, everytime you take the batch out, the order would be different
          dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
          ```

  - **Data Augmentation**
    A common strategy for training neural networks is to introduce randomness in the input data itself. 

    - For example, you can randomly rotate, mirror, scale, and/or crop your images during training.

    - This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.

    - To randomly rotate, scale and crop, then flip your images you would define your transforms like this:

      ```python
      train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], 
                                                                  [0.5, 0.5, 0.5])])
      ```

      You'll also typically want to normalize images with `transforms.Normalize`

      - You pass in a list of means and list of standard deviations, then the color channels 	  are normalized like so:

        ```python
        input[channel] = (input[channel] - mean[channel]) / std[channel]
        ```

        - Subtracting `mean` centers the data around zero and dividing by `std` squishes the values to be between -1 and 1.
        - Normalizing helps keep the network weights near zero which in turn makes backpropagation more stable.
        - Without normalization, networks will tend to fail to learn.

    - Full example:

      ```python
      data_dir = 'Cat_Dog_data'
      
      # TODO: Define transforms for the training data and testing data
      train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor()]) 
      
      test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])
      
      
      # Pass transforms in here, then run the next cell to see how the transforms look
      train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
      test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
      
      trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
      testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
      ```

      - Size always = 224 at the very last before training
      - HorizontalFlip = mirror image

- Train neural network

## 17. Transfer Learning

- Goal

  - how to use pre-trained networks to solved challenging problems in computer vision. (6 architectures)
    ![transfer_learning_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/transfer_learning_1.jpg)

    ![transfer_learning_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/transfer_learning_2.jpg)

          - AlexNet gives us the top one error and the top five error
          - the number stands for the number of layers in this model
          - most of these pretrained model require a 224 by 224 image as the input
          - **Tradeoff: The larger the model, the higher accuracy you can get, the longer it's going to take to compute your predictions and to train and all that**

  - Specifically, you'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html).

    - ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers.
    - Resources: A friendly introduction to Convolutional Neural Networks and Image Recognition
      - https://www.youtube.com/watch?v=2-Ol7ZB0MmU

  - Once trained, these models work astonishingly well as feature detectors for images they weren't trained on.

- What is transfer learning?

  - **Using a pre-trained network on images not in the training set is called transfer learning.** 

- How to use the pre-trained network?

  - **With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now.**

    ```python
    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'
    
    import matplotlib.pyplot as plt
    
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    # torchvision.models
    from torchvision import datasets, transforms, models
    ```

  - Match a normalization used when these models were trained on ImageNet

    - when train this model, each color channel and images were normalized separately
      - Example
        - means are `[0.485,0.456,0.406]`
        - standard deviations are `[0.229, 0.224, 0.225]`

  - For example: load in a model such as DenseNet

    ```python
    model = models.densenet121(pretrained=True)
    model
    ```

    ![transfer_learning_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/transfer_learning_3.jpg)

    The model is built out of two main parts

    1. Features

    - stack of convolutional layers and overall works as a feature detector that can be fed into a classifier

    2. Classifier

    - it is a single fully-connected layer (classifier): Linear(in_features=1024, out_features=1000)

      - **This layer was trained on the ImageNet dataset, so it won't work for our specific problem.** 
      - **That means we need to replace the classifier, but the features will work perfectly on their own**

    -  **In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers.**

      ```python
      # Freeze parameters so we don't backprop through them
      for param in model.parameters():
          param.requires_grad = False
      
      from collections import OrderedDict
      classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 500)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(500, 2)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
          
      model.classifier = classifier
      ```

    - **How do we compute really deep neural network?**

      - Use the GPU to do the calculations

        - The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds.
        - It's also possible to train on multiple GPUs, further decreasing training time.

      - Pytorch uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU.

        - In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`.

        - You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. 

        - Operations

          ```python
          # move them to GPU
          model.cuda()
          images.cuda()
          # move them to CPU
          model.cpu()
          images.cpu()
          ```

          ```python
          # 1st way to compare the computation time
          for cuda in [False, True]:
              
              criterion = nn.NLLLoss()
              optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
              
              if cuda:
                  model.cuda()
              else:
                  model.cpu()
              
              for ii, (inputs, labels) in enumerate(trainloader):
          		inputs, labels = Variable(inputs), Variable(labels)
                  
                  if cuda:
                      inputs, labels = inputs.cuda(), labels.cuda()
          
                  start = time.time()
          
                  outputs = model.forward(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
          
                  if ii==3:
                      break
                  
              print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
              
          ```



          ```python
          # 2nd way (better) Testing the computation time
          import time
          
          for device in ['cpu', 'cuda']:
          
              criterion = nn.NLLLoss()
              # Only train the classifier parameters, feature parameters are frozen
              optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
          
              model.to(device)
          
              for ii, (inputs, labels) in enumerate(trainloader):
          
                  # Move input and label tensors to the GPU
                  inputs, labels = inputs.to(device), labels.to(device)
          
                  start = time.time()
          
                  outputs = model.forward(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
          
                  if ii==3:
                      break
                  
              print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
          ```

        - **You can write device agnostic code which will automatically use CUDA if it's enabled like so:**

          ```python
          # at beginning of the script
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          
          ...
          
          # then whenever you get a new Tensor or Module
          # this won't copy if they are already on the desired device
          input = data.to(device)
          model = MyModule(...).to(device)
          ```

        - Full example

          ```python
          # Use GPU if it's available
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          
          # we can print the model to see the architecture after running this
          model = models.densenet121(pretrained=True)
          
          
          # Freeze parameters so we don't backprop through them
          for param in model.parameters():
              param.requires_grad = False
          
          # Define our new classifier
          classifier = nn.Sequential(nn.Linear(1024, 256),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(256, 2),
                                           nn.LogSoftmax(dim=1))
          
          # print out the model again
          model.fc = classifier
          
          criterion = nn.NLLLoss()
          
          # Only train the classifier parameters, feature parameters are frozen
          optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
          
          model.to(device);
          ```

           Training Step:

          ```python
          # Alternative way (in the video)
          
          
          epochs = 1
          steps = 0
          running_loss = 0
          print_every = 5
          for epoch in range(epochs):
              for inputs, labels in trainloader:
                  steps += 1
                  # Move input and label tensors to the default device
                  inputs, labels = inputs.to(device), labels.to(device)
                  
                  optimizer.zero_grad()
                  
                  logps = model.forward(inputs)
                  loss = criterion(logps, labels)
                  loss.backward()
                  optimizer.step()
          
                  running_loss += loss.item()
                  
                  for steps %% print_every == 0:
                      model.eval()
                      test_loss = 0
                      accuracy = 0
                      
                      
                      for inputs, labels in testloader:
                          logps = model(inputs)
                          loss = criterion(logps, labels)
                          test_loss += batch_loss.item()
                              
                          # Calculate accuracy
                          ps = torch.exp(logps)
                          top_p, top_class = ps.topk(1, dim=1)
                          equals = top_class == labels.view(*top_class.shape)
                          accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                              
                      print(f"Epoch {epoch+1}/{epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Test loss: {test_loss/len(testloader):.3f}.. "
                            f"Test accuracy: {accuracy/len(testloader):.3f}")
                      running_loss = 0
                      model.train()
          ```



          ```python
          
          epochs = 1
          steps = 0
          running_loss = 0
          print_every = 5
          for epoch in range(epochs):
              for inputs, labels in trainloader:
                  steps += 1
                  # Move input and label tensors to the default device
                  inputs, labels = inputs.to(device), labels.to(device)
                  
                  optimizer.zero_grad()
                  
                  logps = model.forward(inputs)
                  loss = criterion(logps, labels)
                  loss.backward()
                  optimizer.step()
          
                  running_loss += loss.item()
                  
                  if steps % print_every == 0:
                      test_loss = 0
                      accuracy = 0
                      model.eval()
                      with torch.no_grad():
                          for inputs, labels in testloader:
                              inputs, labels = inputs.to(device), labels.to(device)
                              logps = model.forward(inputs)
                              batch_loss = criterion(logps, labels)
                              
                              test_loss += batch_loss.item()
                              
                              # Calculate accuracy
                              ps = torch.exp(logps)
                              top_p, top_class = ps.topk(1, dim=1)
                              equals = top_class == labels.view(*top_class.shape)
                              accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                              
                      print(f"Epoch {epoch+1}/{epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Test loss: {test_loss/len(testloader):.3f}.. "
                            f"Test accuracy: {accuracy/len(testloader):.3f}")
                      running_loss = 0
                      model.train()
          ```


## 18. Tips, Tricks and Other Notes

### a) Watch those shapes

In general, you'll want to check that the tensors going through your model and other code are the correct shapes. Make use of the `.shape` method during debugging and development.

### b) A few things to check if your network isn't training appropriately

Make sure you're clearing the gradients in the training loop with `optimizer.zero_grad()`. If you're doing a validation loop, be sure to set the network to evaluation mode with `model.eval()`, then back to training mode with `model.train()`.

### c) CUDA errors

Sometimes you'll see this error:

```
RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #1 ‘mat1’
```

You'll notice the second type is `torch.cuda.FloatTensor`, this means it's a tensor that has been moved to the GPU. It's expecting a tensor with type `torch.FloatTensor`, no `.cuda` there, which means the tensor should be on the CPU. PyTorch can only perform operations on tensors that are on the same device, so either both CPU or both GPU. If you're trying to run your network on the GPU, check to make sure you've moved the model and all necessary tensors to the GPU with `.to(device)` where `device` is either `"cuda"` or `"cpu"`.

