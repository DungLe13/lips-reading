## Automated Lip Reading

Lip reading, also known as audio-visual recognition, has been considered as a solution for speech recognition tasks, especially when the audio is corrupted or when the conversation happened in noisy environments. It can also be an extremely helpful tool for people who are hearing-impaired to communicate through video calls. This task, however, is challenging, due to factors such as the variances in the inputs (facial features, skin colors, speaking speeds, etc.) and the one-to-many relationships between viseme and phoneme. This project aims to tackle lip reading by modeling an agent that is capable of learning the features by interacting with the environment using reinforcement learning methodology. 

### Model

Three dominant components of the model are:

- **Convolutional Neural Network**: VGG16 model pre-trained on ImageNet dataset was used to transform images of the lips region to its vector representation.

- **Long Short-Term Memory network**: a recurrent neural network with long short-term memory cells acts as an agent that used REINFORCE to learn its parameters.

- **Reinforcement Learning**:

	The *components of RL* in Lip Reading setting:
	- An **agent**: the generative model (RNN with LSTMs)
	- An **environment** contains the words and the context vector it sees as input at every time step.
	- A **policy**: the parameters of the generative model
	- An **action** refers to  predicting the next word in the sequence at each time step.
	- A **reward function**: BLEU - evaluating the similarity between the generated text and the ground truth.

![Visualization of Lip Reading Model](https://github.com/DungLe13/lips-reading/blob/master/model.png)

### Reinforcement Learning Methods

The three RL methods implemented on this project are:

| Method | References | Implementation |
|---|---|---|
| REINFORCE | Williams, 1992; Zaremba & Sutskever, 2015 | [REINFORCE](https://github.com/DungLe13/lips-reading/blob/master/Policy%20Gradient/REINFORCE.py) |
| Deep Q-Network | Mnih et al., 2014 | [DDQN](https://github.com/DungLe13/lips-reading/blob/master/DDQN/DDQN.py) |
| Asynchronous Advantage Actor Critic (A3C) | Mnih et al., 2016 | [A3C](https://github.com/DungLe13/lips-reading/blob/master/Actor%20Critic/A3C.py) |

### Important Notes

1. The GRID corpus contains 33,000 facial recordings. In the folder `GRID corpus/vectors`, only 100 vector representations of 100 videos are shown to demonstrate the method. The GRID corpus can be found [HERE](http://spandh.dcs.shef.ac.uk/gridcorpus/).
2. Another dataset that I plan to use in this project is the BBC-Oxford 'Multi-View Lip Reading Sentences' (MV-LRS) Dataset, which can be found [HERE](http://www.robots.ox.ac.uk/~vgg/data/lip_reading_sentences/).
3. In folder `Video Processing`, the file `shape_predictor_68_face_landmarks.dat`, which was used to detect lips region, is left out due to its large size. This file can be downloaded [HERE](https://drive.google.com/file/d/1-8zm3bu9Y5kvjAS6C7tUlpYeJOVmxPZe/view?usp=sharing).
4. In folder `CNN/pretrained-VGG16`, the file `vgg16_weights.npz` is also left out due to its large size. This file contains the weights pre-trained using ImageNet dataset. This file can be downloaded [HERE](http://www.cs.toronto.edu/~frossard/post/vgg16/).