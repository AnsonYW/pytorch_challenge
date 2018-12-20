# Convolutional Neural Networks

[TOC]

## Applicaiton of CNNs

- Voice User interfaces

  - Google released WaveNet Model (link)
    - Take any text as input and return computer-generated audio of a human reading in the text
    - if you supply the algorithm with enough samples of your voice, it is possible train it sound like you
    - researchers used a variant of the WaveNet model to generate songs

- Natural Language Processing

  - RNN in general is used more than CNN
  - extract information from sentences
    - this information can be used to classify <u>sentiment</u>
      - is the writer happy or sad
      - if talking about a movie, did they like or dislike it

- computer vision

  - image classification tasks (focus of this video)
    - Given an image, your CNN will assign a corresponding label which you believe summarizes the content of the image
    - Wide applications
      - this teaches AI agents to play video games such as <u>Atari Breakout</u>
        - CNN-based mode are able to learn to play games without being given any prior knowledge of what a ball is, and without even being told precisely what the controls do
        - The agent only see the screen and its score, but it does have access of the control that you would give a human user
        - With these simple information, CNNs can extract crucial information that allows them to develop a useful strategy
      - CNNs have even been trained to play Pictionary
        - Quick draw
          - it guesses what you are drawing based on your finger-drawn picture
      - Go: ancient Chinese board game: considered one of the most complex games in existence
        - it has more configurations in the game than there are atoms in the universe
        - DeepMind: train AI agent to beat human professional Go players
      - Allows drones to navigate unfamiliar territory
        - drone are now used to deliver medical supplies to remote areas
          - CNNs give the drone the ability to see or to determine what's happening in streaming video data
      - Decode images of text (extract text from picture)
        - digitalize the historical book or your handwritten notes
          - we can identify images of letters or numbers or punctuation
          - we can aid self-driving caars with reading road signs
            - Google has built a better more accurate street maps of the world by training an algorithm that can read house numbers sign from street view images
      - In general, CNNs is the state of the art in a lot of cases

- ### Optional Resources

  - Read about the [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) model.
    - Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found [here](http://www.creativeai.net/posts/W2C3baXvf2yJSLbY6/a-neural-parametric-singing-synthesizer).
  - Learn about CNNs [for text classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
    - You might like to sign up for the author's [Deep Learning Newsletter](https://www.getrevue.co/profile/wildml)!
  - Read about Facebook's [novel CNN approach](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/) for language translation that achieves state-of-the-art accuracy at nine times the speed of RNN models.

  - Play [Atari games](https://deepmind.com/research/dqn/) with a CNN and reinforcement learning. You can [download](https://sites.google.com/a/deepmind.com/dqn/)the code that comes with this paper.
    - If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).
  - Play [pictionary](https://quickdraw.withgoogle.com/#) with a CNN!
    - Also check out all of the other cool implementations on the [A.I. Experiments](https://aiexperiments.withgoogle.com/) website. Be sure not to miss [AutoDraw](https://www.autodraw.com/)!
  - Read more about [AlphaGo](https://deepmind.com/research/alphago/).
    - Check out [this article](https://www.technologyreview.com/s/604273/finding-solace-in-defeat-by-artificial-intelligence/?set=604287), which asks the question: *If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?*
  - Check out these *really cool* videos with drones that are powered by CNNs.
    - Here's an interview with a startup - [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y).
    - Outdoor autonomous navigation is typically accomplished through the use of the [global positioning system (GPS)](http://www.droneomega.com/gps-drone-navigation-works/), but here's a demo with a CNN-powered [autonomous drone](https://www.youtube.com/watch?v=wSFYOw4VIYY).
  - If you're excited about using CNNs in self-driving cars, you're encouraged to check out:
    - our [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), where we classify signs in the [German Traffic Sign](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset in [this project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project).
    - our [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009), where we classify house numbers from the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) dataset in [this project](https://github.com/udacity/machine-learning/tree/master/projects/digit_recognition).
    - this [series of blog posts](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/) that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.
  - Check out some additional applications not mentioned in the video.
    - Some of the world's most famous paintings have been [turned into 3D](http://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1)for the visually impaired. Although the article does not mention *how* this was done, we note that it is possible to use a CNN to [predict depth](https://www.cs.nyu.edu/~deigen/depth/) from a single image.
    - Check out [this research](https://research.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
    - CNNs are used to [save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
    - An app called [FaceApp](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.

## Lesson Outline

- Improve our ability to classify images

- CNN can look into the picture as a whole, learn to identity <u>spatial pattern</u>
  ![lesson_outline_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/lesson_outline_1.jpg)
  - such as prominent colors and shapes
  - or whether the texture is fuzzy or smooth
- The shapes and colors that define any image and in the object in a image are often called features
- CNN is able to learn this features

<u>What is a feature?</u>

- think of what we are visually drawn to when we first see an object and when we identify different objects
  - e.g. what do we look at to distinguish a cat and a dog
    - the shape of the eyes, the size, and how they move...
  - e.g. we see a person walking toward us and we want to see if it is someone we know
    - we may look at their face, even further their general shape
      - the distinct shape of a person and their eye color are great examples of distingusing features

## MNIST Dataset

- How to recognize a single object in a image
  ![image-20181218210435013](/Users/anson/Library/Application Support/typora-user-images/image-20181218210435013.png)

- We want to design an image classifier
  - input: image of a hand-written number
  - output: a class of the number
- How to build this
  - use a mnist database
    ![mnist_dataset_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/mnist_dataset_2.jpg)
- Difficulties in the dataset
  - some "3"s will be recognized as 8
  - some "4" can be recognized as 9
- Deep learning: data-driven approach, train an algorithm that can examine these images and discover patterns that distinguish one number from another
- The algorithm needs to attain some level of understanding of what makes a handdrawn 1 look like a 1
  - and how image of 1 different from image of 2

## How Computers Interpret Images

![computer_interpret_images_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/computer_interpret_images_1.jpg)

- images is interpreted as an <u>array</u>
- each box in the image is called a pixel, and it has a value
- MNIST database has pictues with 28 pixels high and 28 pixels wide = 28 x 28 array
- white is encoded as 255, black is encoded as 0

![computer_interpret_images_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/computer_interpret_images_2.jpg)

- The image is then passed for a pre-processing step
  - <u>normalization</u>: divide all pixel value with 255
  - range 0 -1 instead of 0 - 255
  - help our algorithm train better
    - rely on gradient calculation
    - learn how important/how weighty a pixel should be determining the class of the image
    - make the gradient calculation more consistent and not get so large to slow down and prevent a network from training
    - **Why normalization?**
      - Ensures that each input (each pixel value, in this case) comes from a standard distribution
      - The range of pixel values in one input image are the same as the range in another image.
      - This standardization makes our model train and reach a minimum error, faster
      - Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. [Read more about the Normalize transformation in PyTorch.](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)
      - The distribution of such data should resemble a [Gaussian function](http://mathworld.wolfram.com/GaussianFunction.html) centered at zero. 
        - For image inputs we need the pixel numbers to be positive, so we often choose to scale the data in a normalized range [0,1]
- How to classify a image using MLP? **(as it only receives vector as input)**

![computer_interpret_images_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/computer_interpret_images_3.jpg)

- convert any image of array to vector
- 4 x 4 matrix
- vector of 16 entries

![computer_interpret_images_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/computer_interpret_images_4.jpg)

- The vector can be fed as the input of the MLP

## MLP Structure & Class Scores

<u>Create a neural network for discovering the patterns in our data</u>

After training, our model should be able to look at the images that haven't trained on, which is called the <u>test data</u>

![MLP_structure_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/MLP_structure_1.jpg)

- Input layer:
  - images which are converted into vectors with 784 entries, therefore we have 784 nodes 
- Output layer:
  - 10 different numbers --> so we have 10 nodes --> 10 output values for each of the class

![MLP_structure_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/MLP_structure_2.jpg)

- Output values are often called as "class scores"
  - high scores: the model is very certain that a image falls into certain class
    - e.g. input = 3
      - In the output layer, has high score in class 3 and low score in other classes, 8 may get pretty high scores as 8 looks like 3 in some cases

![MLP_structure_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/MLP_structure_3.jpg)

- output layers can be represented as <u>scores</u> or <u>bar chart</u>
  - indicating the relative strength of the scores

![MLP_structure_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/MLP_structure_4.jpg)

- The things up to we define is inbetween the input and output layers
  - How many hidden layers we want to include?
  - How many nodes we want to have for each hidden layer?

![MLP_structure_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/MLP_structure_5.jpg)

- usually look at the papers or related work as a good guide
- search for
  - MLP for MNIST
  - MLP for small scale grey images

## Do Your Research

- Look into google and find reputable source **(lower case may help me to find better result)**
  - keras + Github (example of good source)
- Look at the code

![do_research_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/do_research_1.jpg)

- Import mnist dataset

![do_research_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/do_research_3.jpg)

- reshape the image

![do_research_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/do_research_2.jpg)

- look at the model
  - <u>**2 layers**</u> with relu activation function
  - dropout inbetween
  - output = number of classes--> softmax

![do_research_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/do_research_4.jpg)

- When we look at the source like this, we try to evaluate whether it makes sense
  - more layers, more complex, but we do not want to be too complex
  - small images --> 2 layers should be fine

![do_research_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/do_research_5.jpg)

- Keep looking at more sources, and see whether we can find a better one and test it in code

## Loss & Optimization

![loss_and_optimization_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_1.jpg)

- input = image 2, ouput = different scores for different classes
  - the higher the score, the more certain the image is likely to fall into this category
    - e.g. 8 is the largest (most likely) and 3 is the smallest (least likely)
- However, it is incorrect, we need to learn from mistakes

![loss_and_optimization_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_2.jpg)

- Backpropagation: compute the gradient of the loss with respect to the model's weight
  - find out which weight is responsible for any errors
- Optimization function (e.g. gradient descent)

![loss_and_optimization_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_3.jpg)

- make this output value more interpretable
  - apply softmax activation function

![loss_and_optimization_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_4.jpg)

- then we will get the 10 values bounded in the green box
- our goal is to update the weight of the network and respond to the mistakes, so next time the model will predict 2 is most likely to be the predicted class
- We then need to measure how far our current model is from perfection
  - Loss function
    - we will use categorical cross entropy loss (multi-class classification)
    - take the negative log loss of the value = 1.82
  - backpropagation will find out which weight is responsible for each error

![loss_and_optimization_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_5.jpg)

- we get a better value when the prediction is better

![loss_and_optimization_6](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_6.jpg)

- the goal is to minimize the loss function, therefore it can give us the most accurate predictions

![loss_and_optimization_7](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_7.jpg)

- we need to find a way to decent to the lowest value, which is the road of an optimizer

![loss_and_optimization_8](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/loss_and_optimization_8.jpg)

- there are a number of ways to perform gradient descent
- each method has a corresponding optimizer
- All of them are racing towards the minimum of the function
  - encourage to test all of them in your code

## Defining a Network in PyTorch

```python
# import libraries
import torch
import numpy as np
```

Load and visualize the data

```python
# import the torch dataset
from torchvision import datasets 
# import the transformation libraries
import torchvision.transforms as transforms

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor() 
# Tensor is a similar thing as array, only it can be put into GPU calculation

# choose the training and test datasets
# loading data, download it and transform it
# download the data into a directory called data
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
```

- batch_size: number of images will be seen in one iteration
  - one iteration means one time that a network makes a mistake and learn from them using backpropagation
- num_workers: if you want to run in parallel, for most cases, zero works fine here
- The data loader takes in the data we define above, batch size and numbr of workers
  - it allows to iterate the data one batch at a time

<u>First step in any image classification task: Visualize a Batch of Training Data</u>

![define_network_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/define_network_2.jpg)

```python
import matplotlib.pyplot as plt
%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20): # plot 20 of them
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
```

View an Image in More detail
![define_network_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/define_network_3.jpg)

```python
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
```

Defining the Network Architecture (to do)

```python
import torch.nn as nn
import torch.nn.functional as F

## TODO: Define the NN architecture
class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512 # optional move
        hidden_2 = 512
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, 1) # 784 entrie, 1 = number of ouput has to 			be changed
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self. dropout(x)
        # add hidden layer, with relu activation function
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)
```

Important:

1. init function: 
   - to define any neural network in Pytorch,we need to define names and any layers that have learnt weight
     - e.g. in this case: any linear layers you define
2. feedforward behavior
   - x will be passed through the network and transformed
   - flatten first into desired shape
   - first full connected layer, named as fc1
   - and then add a <u>Relu</u>
     - The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values, this step ensures that our model trains efficiently
     - A ReLu activation function stands for "rectified linear unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the **positive** part of the input, `x`. So, for an input image with any *negative* pixel values, this would turn all those values to `0`, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.

## Training the Network

It's recommended that you use cross-entropy loss for classification

![train_network_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/train_network_1.jpg)

softmax to the output and apply negative log

 - it means you only need class scores

![train_network_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/train_network_2.jpg)

- The losses are averaged across observation for each minibatch
  - each batch size = 20, the losses of each training will be the average of 20

documentation of [Loss Function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [Optimizer](http://pytorch.org/docs/stable/optim.html)

```python
## TODO: Specify loss and optimization functions

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Train the network

```python
# number of epochs to train the model
n_epochs = 30  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
```

- n_epochs: means how many times will you go through the dataset
  - small to large, keep track of the running loss
- inside the epoch loop is the batch loop
- train loader
  - we can look at the data and its label in that batch
- Clear out any gradient calculation that pytorch has accumulated 
  - `.zero_grad()`
- forward pass
  - mode(data)
  - output = scores
- loss function -- > calculate the entropy loss
  - compare the ouput and the true label
- backpropagation, sinple optimization step
- running trainig loss (accumulated loss)
  - since the function return the average, so we need to multiply by the batch size
- After the batch loop, calculate the average loss of the epoch
  - divide the accumulate loss by total number of images in the training set
- and we will print it accumulate loss

Test

```python
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

Test the trained network

`model.eval(`) will set all the layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation!

```python
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

>>output:
    Test Loss: 0.055157

Test Accuracy of     0: 99% (971/980)
Test Accuracy of     1: 99% (1125/1135)
Test Accuracy of     2: 98% (1014/1032)
Test Accuracy of     3: 98% (994/1010)
Test Accuracy of     4: 98% (963/982)
Test Accuracy of     5: 98% (877/892)
Test Accuracy of     6: 98% (940/958)
Test Accuracy of     7: 97% (1006/1028)
Test Accuracy of     8: 97% (946/974)
Test Accuracy of     9: 97% (988/1009)

Test Accuracy (Overall): 98% (9824/10000)
```

Visualize Sample Test Results

```python
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))
```

## Model Validation

![model_validation_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_validation_1.jpg)

![model_validation_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_validation_2.jpg)

- The exact epoch to stop training is hard to determine
- Criteria
  - accurate but not overfitting

![model_validation_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_validation_3.jpg)

- Divide the dataset into training, validation and test set
- After training, we look at the loss of both training set and validation set
  - **We do not use any part of the validation set for back propagation**
    - so that it is possible to tell us whether our model can be generalized
  - we try to find all patterns in the training set

![model_validation_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_validation_4.jpg)

- The part the training loss is decreasing and validation loss is increasing
  - Overfitting

![model_validation_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_validation_5.jpg)

If we have multiple architectures to choose from (e.g number of layer)

- save the weight of each architecture
- **pick the model with the lowest validation loss

Why we need to create validation instead of simply using the test set?

- Because in the test set, we are checking whether the model is able to generalize for the trully unseen data

## Validation Loss

<u>What should be added to the code?</u>

1. Load and Visualize the Data

   ```python
   from torchvision import datasets
   import torchvision.transforms as transforms
   from torch.utils.data.sampler import SubsetRandomSampler # split the training data --------> (CHANGED)
   
   # number of subprocesses to use for data loading
   num_workers = 0
   # how many samples per batch to load
   batch_size = 20
   # percentage of training set to use as validation # ----------> (CHANGED)
   valid_size = 0.2 
   
   # convert data to torch.FloatTensor
   transform = transforms.ToTensor()
   
   # choose the training and test datasets
   train_data = datasets.MNIST(root='data', train=True,
                                      download=True, transform=transform)
   test_data = datasets.MNIST(root='data', train=False,
                                     download=True, transform=transform)
   
   # obtain training indices that will be used for validation
   num_train = len(train_data) # -----> how many training images (CHANGED)
   indices = list(range(num_train)) # List out all indices ------> (CHANGED)
   np.random.shuffle(indices) # shuffle ---------> (CHANGED)
   split = int(np.floor(valid_size * num_train)) # --->Split (CHANGED)
   train_idx, valid_idx = indices[split:], indices[:split] # -> Split (CHANGED)
   
   # define samplers for obtaining training and validation batches
   train_sampler = SubsetRandomSampler(train_idx) # -----> CHANGED
   valid_sampler = SubsetRandomSampler(valid_idx) # -----> CHANGED
   
   # prepare data loaders
   train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
       sampler=train_sampler, num_workers=num_workers) # add sampler argument (CHANGED)
   valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
       sampler=valid_sampler, num_workers=num_workers) # add validation dataset loader
   test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
       num_workers=num_workers)
   ```

2. Train the network

   ```python
   # number of epochs to train the model
   n_epochs = 50
   
   # initialize tracker for minimum validation loss
   valid_loss_min = np.Inf # set initial "min" to infinity ----> check the change of validation loss, guarantee it will update after the first epoch(CHANGED)
   
   for epoch in range(n_epochs):
       # monitor training loss
       train_loss = 0.0
       valid_loss = 0.0
       
       ###################
       # train the model #
       ###################
       model.train() # prep model for training
       for data, target in train_loader:
           # clear the gradients of all optimized variables
           optimizer.zero_grad()
           # forward pass: compute predicted outputs by passing inputs to the model
           output = model(data)
           # calculate the loss
           loss = criterion(output, target)
           # backward pass: compute gradient of the loss with respect to model parameters
           loss.backward()
           # perform a single optimization step (parameter update)
           optimizer.step()
           # update running training loss
           train_loss += loss.item()*data.size(0)
           
       ######################    
       # validate the model # -----> validation batch loop (CHANGED)
       ######################
       model.eval() # prep model for evaluation
       for data, target in valid_loader:
           # forward pass: compute predicted outputs by passing inputs to the model
           output = model(data)
           # calculate the loss
           loss = criterion(output, target)
           # update running validation loss 
           valid_loss += loss.item()*data.size(0)
           
       # print training/validation statistics 
       # calculate average loss over an epoch
       train_loss = train_loss/len(train_loader.dataset)
       valid_loss = valid_loss/len(valid_loader.dataset)
       
       print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
           epoch+1, 
           train_loss,
           valid_loss
           )) # also print the validation loss (CHANGED)
       
       # save model if validation loss has decreased
       if valid_loss <= valid_loss_min:
           print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
           valid_loss_min,
           valid_loss))
           torch.save(model.state_dict(), 'model.pt')
           valid_loss_min = valid_loss # store the model when there is a new minimum as model.pt ---> (CHANGED)
   ```

   ```python
   
   Validation loss decreased (0.017454 --> 0.016991).  Saving model ...
   Epoch: 26 	Training Loss: 0.030163 	Validation Loss: 0.016554
   Validation loss decreased (0.016991 --> 0.016554).  Saving model ...
   Epoch: 27 	Training Loss: 0.028369 	Validation Loss: 0.017595
   Epoch: 28 	Training Loss: 0.026245 	Validation Loss: 0.016682
   Epoch: 29 	Training Loss: 0.025983 	Validation Loss: 0.017080
   Epoch: 30 	Training Loss: 0.024357 	Validation Loss: 0.016169
   Validation loss decreased (0.016554 --> 0.016169).  Saving model ...
   Epoch: 31 	Training Loss: 0.022118 	Validation Loss: 0.016334
   Epoch: 32 	Training Loss: 0.023228 	Validation Loss: 0.016612
   Epoch: 33 	Training Loss: 0.020928 	Validation Loss: 0.016693
   Epoch: 34 	Training Loss: 0.019909 	Validation Loss: 0.016322
   Epoch: 35 	Training Loss: 0.018557 	Validation Loss: 0.016833
   Epoch: 36 	Training Loss: 0.018037 	Validation Loss: 0.016070
   Validation loss decreased (0.016169 --> 0.016070).  Saving model ...
   Epoch: 37 	Training Loss: 0.017053 	Validation Loss: 0.015298 
   Validation loss decreased (0.016070 --> 0.015298).  Saving model ... # after this, the loss is more or less similar
   Epoch: 38 	Training Loss: 0.016680 	Validation Loss: 0.016685
   Epoch: 39 	Training Loss: 0.015662 	Validation Loss: 0.016136
   Epoch: 40 	Training Loss: 0.015871 	Validation Loss: 0.016163
   ```

3. Load the Model with Lowest Validation Loss

   ```python
   model.load_state_dict(torch.load('model.pt')) # CHANGED
   ```

   - The result is similar without validation, as the loss doesn't change much
     - the images in this case is very similar, very preprocessed, the digit look pretty much the same
   - Validation step will become more important when there is a higher variety of the dataset

## Image Classification Steps

<u>Full pipeline</u>

![model_classification_step](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/model_classification_step.jpg)

## MLPs vs CNNs

- CNN usually perform so much better than MLP in the real world dataset (they cannot be compared)

- highest accuarcy in MNIST

  - ![mlp_vs_cnn_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/mlp_vs_cnn_1.jpg)

    - MLP performs very well in the very organized dataset like MNIST

    ![mlp_vs_cnn_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/mlp_vs_cnn_2.jpg)

    - there should be situation in which data does not lie in the middle of the grid, it can be small and large
    - In these cases, CNN truely shines
    - Why?

    ![mlp_vs_cnn_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/mlp_vs_cnn_3.jpg)

    - feed image to MLP
    - first convert the image into vector
      - just a simple structure of numbers with no special structure
      - it has no knowledge in that fact that the number is spatially arranged in a grid
    - CNN are builiding on recognizing multidimentional data
    - CNN understands image pixel that are close together are mostly related than the ones far apart

## Local Connectivity - towards motivating and defining CNNs

![local_connectivity_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_1.jpg)

| MLP                                                          | CNNs                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| - use a lot of parameters, for 28x28 images, already contained over half a milion parameters<br />- computational complexity of moderately sized imags could get out of control easily<br />- Throw away all of the 2D information contained in an image when we flatten its matrix into a vector | - Yet, this spatial information or knowledge where the pixels are located in reference ot each other is relevant to understanding the image and could aid significantly towards elucidating the patterns contained in the pixel values.<br/>- That is why we need an entirely new way of processing image input, where the 2-D information is not entirely lost<br />- features:<br />1. connections between layers are informed by the 2-D structure of the image matrix<br />2. accept matrix as input |



![local_connectivity_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_2.jpg)

- Example: 4 x 4 images of handwritten digits
- goal is to classify the digit that's depicted in the image
- Right to left
  - 4 x 4 matrix has been converted to a 16 dimensional vector
  - MLP
    - input: 16 dimension vector
    - middle: single hidden layer with 4 nodes
    - output: softmax activation function and returns a 10-dimensional vector
      - it then contains the probability that the image depicts each of the possible digits zero through nine
        - if the model is good --> predict a seven most probably
        - simplication as follows

![local_connectivity_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_3.jpg)

- There may be some redundancy
  - **does every hidden node need to be connected to every pixel in the original image**
  - perhaps not, we can break the image into four regions

![local_connectivity_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_4.jpg)

- four regions include 1) red, 2) green, 3) yellow and 4) blue
- each hidden node could be connected to only the pixels in one of these four regions
- Here, each headed node sees only a quarter of the original image

![local_connectivity_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_5.jpg)

- again comparing with the previous mentioned fully-connected layer, an understanding of the entire image all at once
- the new regional breakdown and the assignment of small local groups of pixels to different hidden nodes, every hidden node finds patterns in only one of the four region in the image
- Then, each hidden node still reports to the output layer where the output layer combines the findings for the discovered patterns learned separately in each region
- The above is called locally connected layer
  - it uses far fewer parameters than a densely connected layer
  - It is then less prone to overfitting and truly understands how to tease out the patterns contained in image data.

![local_connectivity_9](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_9.jpg)

![local_connectivity_6](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_6.jpg)

- **we can rearrange each of these vectors as a matrix**
  - The relationships between the nodes in each layer are more obvious
  - We could expand the number of patterns that we are able to detect while still making use of the 2-D structure to selectively and conversatively add weights to the model by introducing more hidden nodes, where each is still confined to analyzing a single small region within the image
  - After all, by expanding the number nodes in the hidden layer, we can discover more complex patterns in our data
  - As shown in the above, we now have two collections of hidden nodes where each collection contains nodes responsible for examing a different region of the image.

![local_connectivity_10](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_10.jpg)

- it will prove useful to have each of the hidden nodes within a collection share a common group of weights
- The idea being that different regions within the image can share the same kind of information
  - it means evey pattern that's relevant towards understanding the image could appear anywhere within the image

![local_connectivity_8](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/local_connectivity_8.jpg)

- Example
  - you want your network to say it's an image containing a cat
    - it does not matter where the cat is
      - if your network need to learn the cat located on the left corner or the right corner independently, that's a lot of work that it has to do
    - Instead, we tell the network explicity that objects and images are largely the same whether they are on the left or the right of the picture
      - this is partially accomplished through weight sharing

## Filters and the Convolutional Layer

- CNN is a special kind of neural network that it can remember spacial information
- The neural networks mentioned before only look at the individual input
- CNN can look at the image as a whole or in batches and analyze in groups of pixels in a time

![filters_CNN_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/filters_CNN_1.jpg)

- The key to preserve spacial information is the convolutional layer
- convolutionay layer applies a series of image filters, also known as convolutional kernels

![filters_CNN_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/filters_CNN_2.jpg)

- The resulting filtered images have different experiences
- The filters may have extracted features such as
  - edges of object
  - the color that is distinguished from different classes

![filters_CNN_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/filters_CNN_3.jpg)

​![filters_CNN_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/filters_CNN_4.jpg)

- spacial information such as the curves and lines that a 6 has to distinguish from other numbers
- Later layers learn combine different colors and spacial features to produce an output like a class signal

## Filtes & Edges

- When we are talking about spacial pattern of an image
  - color or shape
- Shape
  - it can thought of as pattern of intensity of an image
    - intensity is a measure of dark and light similar to brightness
      - we use this to detect the shape of an object
        - e.g. distinguish a person from a background in a image
          - we can look at the contrast that occurs at the person ends and the background begins. to define the boundary that separate the two
          - we can often identify the edges of an image by looking an abrupt change of intensity (image from very dark to white area)
  - image filter that looks at groups of pixels and detect big changes in intensity in an image
  - The output shows the edges of objects and differing textures
    - show various edges and shapes

## Frequency in images

- For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s ([Hz](https://en.wikipedia.org/wiki/Hertz)), and high pitches and made by high-frequency waves
-  frequency in images is a **rate of change**
  -  images change in space, and a high frequency image is one where the intensity changes a lot.
  - And the level of brightness changes quickly from one pixel to the next.
  - A low frequency image may be one that is relatively uniform in brightness or changes very slowly
- Most images have both high-frequency and low-frequency components. 
  - In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another.
  - igher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.
  - **High-frequency components also correspond to the edges of objects in images**, which can help us classify those objects.

## High-pass Filters

![high_pass_filters_1](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_1.jpg)

- In image processin, filters are used to filter out unwanted or irrelevant information in an image
- amplify features like object boundaries or other distinguishing traits

![high_pass_filters_2](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_2.jpg)

- High-pass filters are used to make an image appear sharper and enhance high-frequency parts of an image, which are areas where the levels of intensity in neighboring pixel rapidly change like very dark to very light pixels
- Since we are looking at patterns of intensity, the filters we will be working with will be operating on grayscale in greyscale images that represent this information and display paatterns of lightness and darkness format

![behigh_pass_filters_3](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_3.jpg)

- There is no change or a little change in intensity in the original picture,such as in the large areas of dark and light
  - A high-pass filter will black these areas out and the pixels back
  - But in certain areas where a pixel is way brighter than its immediate neighbours, the high-pass filter will enhance that change and create a line
  - we can then see that this has the effect of emphasizing edges

![high_pass_filters_4](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_4.jpg)

- How does it work?
  - It's a 3 x 3 kernel whose elements all sum to zero
  - **It is important that for edge detection all of the elements sum to zero**
  - In this case, subtracting the value of the pixels that surround a certain pixel
  - **if these kernel values did not add up to zero**

![high_pass_filters_5](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_5.jpg)

- positively weighted (brightening)

![high_pass_filters_6](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_6.jpg)

- negatively weighted (darkening the entire filtered image respectively)

![high_pass_filters_7](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_7.jpg)

To apply this fitler, an input image F(xy) is convolved with this kernel

- The convolution is represented by an asterisk, (note that not to be mistaken for a multiplication)
- kernel convolution is an important operation in computer vision applications and it is the basis for convolutional neural networks

![high_pass_filters_8](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_8.jpg)

- It involves taking a kernel, which is our small grid of numbers and passing it over an image pixel by pixel transforming it based on what these numbers are 
- We will see that by changing numbers

![high_pass_filters_9](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_9.jpg)

- To zoom in

![high_pass_filters_10](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_10.jpg)

- For every pixel in this greyscale image, we put our kernel over it so that the **pixel is in the center of the kernel** (220 in this case)

![high_pass_filters_11](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_11.jpg)

- Multiplication
  -  0 * 120
  - -1 * 140
  - 0 * 120
  - -1 * 225
  - 4 * 220
- Sum
  - add all the values up
  - **60 means a very small edge has been detected**
    - because it only changes from light at the bottom to a little dark on top
- **These multiples in our kernel are often called ==weights== because they determine how important or how weighty a pixel is in forming a new output images**
  - in this case, for edge detection
    - the center pixel is the most important
    - followed by its closest pixels on the top and bottom and its left and right, which are negative weights that increase the contrast in the image
    - The corners are the farthest away from the center pixel and in this example, we do not give them any weight

![high_pass_filters_12](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_12.jpg)

- We do this for every pixel position in the original image until you have a complete output image that's about the same size as the input image with new filtered pixel values

![high_pass_filters_13](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/high_pass_filters_13.jpg)

- What we need to concern?

  - weighted sum
  - What to do at the edges and corners of your image since the kernel cannot be nicely laid over 3 x 3 pixel values everywhere

- There are number of ways to process edges (the most common ones are shown as below)

  - **Extend**:
    -  The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.
  - **Padding**:
    -  The image is padded with a border of 0's, black pixels.

  - **Crop**:
    -  Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

## Quiz: Kernels

![kernels](/Users/anson/Desktop/Courses/Year 2_FirstSem/Pytorch Challenge/image/kernels.jpg)

Answer = D

- This kernel finds the difference between the top and bottom edges surrounding a given pixel.

## OpenCV & Creating Custom Filters

## Notebook: Finding Edges

## Convolutional Layer

## Convolutional Layers (Part 2)

## Stride and Padding

## Pooling Layers

## Notebook: Layer Visualization

## Increasing Depth

## CNNs for Image Classification

## Convolutional Layers in Pytorch

## Feature Vector

## CIFAR Classification Example

## Notebook: CNN Classification

## CNNs in Pytorch

## Image Augmentation

## Augmentation Using Transformation

## Groundbreaking CNN Architectures

## Visualizing CNNs

## Summary of CNNs





