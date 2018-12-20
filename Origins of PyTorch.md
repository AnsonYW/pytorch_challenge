# Talking Pytorch with the Creator of Pytorch

[TOC]

## 1. Origins of PyTorch

- Background of creator
  - The creator started as a visual effect artist
  - Spent 6 months with professor's lab
  - Went to CMU tried robotics
  - Landed at NYU and Yann LeCun slab doing deep learning
- Why do you create PyTorch
  - he worked on the project called EB learn which was like two generations before in terms of deep learning framwork
  - It was a hard time
    - took 15 minutes for simply kicking off a program
    - become active to contribute and start helping people
    - Reason
      - Reseach progress changes, tools need to get updated as well
      - experienced a really tough project, calle cocoa detection challlenge
        - it is difficult to implement very compleicated network
- How does it get spreaded?
  - start with people he knew in his own community
  - word of mouth
  - one person --> to entire lab switch from one framework to another framework

## 2. Debugging and Designing PyTorch

<u> Aim of Pytorch</u>

- It should be very imperative, usable, pythonic in the user side but at the same time as fast as any other framework

<u>Impact of the aim</u>

- Large part of Pytorch lives in C++
- Whatever is user facing are still in Python to that you can attach your debugger, print and all that 
- The user land would need to know are in Python but any of the critical parts that are internal as they are all in C++

## 3. From Research to Production

- Pytorch is developed through a reinforcemetn loop with researchers as they developed
- It is really similar to the libraries in NumPy
- **Features that make Pytorch easier to deploy models to production**
  - What do they mean by in production is we want to export the whole model into like C++ runtime or you want to run into 32 bits, 64 bits, 8 bits...
  - It's converting all your Python code inot is it C++ code or some sort of intermediate representation that can be run in C++, then users can run it in their own virtual env
- **When you see like a bug in the intermediate representation, what do you need to do?**
  - Nothing
  - because when we are building our model, we add our function annotation as an optional and we can disable comment our the functionality vision or you can use a global variable to switch off all compilations
  - **90% of the time when you are doing research, regular PyTorch mode**
  - **10% of the time, you think your model is ready to go, you add function annotations and it becomes this magical black box you do not usually need to touch**
  - Remove the annotation when you are doing experiments

## 4. Hybrid Frontend

- Background
  - when you have a giant model, you might not want to change big parts of the model
- What can we do?
  - we can just add function annotations to like the smaller components while changing the other confidence a lot which we called a new programming model hybrid front-end
    - because you can make parts of a model like complied
    - parts of my model like still experimenting
- Can we use optimization to train the model faster?
  - The model is trained, which is powered by JIT compiler
    - its goal is to make sure we can export everything to production ready
  - People's main concern: you cannot ship it to production
    - So now pytorch tend to leverage JIT compiler using for exporting a train model
      - while you are training to make sure things get faster
  - **Introduce a open standard called Onnx**
    - it is a standard for all deep learning framework
      - someone can take a model that is trained in one and export it to another framework
        - So we always export the model to onnx and run it in as another framework
    - Short coming
      - standard has developed as fast enough to cover the most complex model

## 5. Cutting-edge Applications in Pytorch

- paper written by Andy Brock, called smashed
- fair seek project from facebook: text to text processing model
  - give some text and then it would generate another text
  - hierarchical story
    - like I want a story of a boy swimming in a pond

## 6. User Needs and Adding Features

- When they are exploring new ideas, they do not want to be seeing like a 10x drop in performance
  - because some standardize form are already optimized with GPU
    - which is 10 times faster than you hand write the code
- Solution: JIT compiler
- More interactive tutorial based on Python notebooks
- First-class integration with collab, so people can get free GPU
  - GPU, TPU and tensor board are closely collaboraating with Pytorch as well

## 7. PyTorch and the Facebook Product

- support tools for AI primary research
- open source, do AI research in the open and advanced humanity using AI

## 8. The Future of PyTorch

- It can become a very pervasive and essential confident in many other fields
  - because like in healthcare, chemistry, neural science, particle physics subdomain are only doing **unit implementation** instead of full modelling, because they do not equip with deep learning knowledge
- Lower the barrier of entry the use of deep learning
  - let even neuroscientist can just understand what they actually need and then build a cute package
  - Not only make people who use people feel it is pythonic, it should be "pythonic" in all other sub-fields insiders as well (e.g. neuroscience think, physics sake :D)

## 9. Learning More in AI

- Get hands on experience in the first day

