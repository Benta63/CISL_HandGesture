# CISL_HandGesture
As part of an undergraduate research project with the Cognitive Immersive Systems Laboratory, I created (hopefully) a hand gesture recognition software using a Convolutional Neural Network.

##
To run the network, navigate to the `src` folder and run `main.py`. In the future, I may add command line arguments for epochs, learning rate, etc, but for now they are local variables in `main.py`. Additionally, this rendition does not support CUDA. There is however a messy implementation with Google Colabs with CUDA support (https://colab.research.google.com/drive/1_UB02owkSjffu4JtgoZjjHy3BH8SBBwm). 

## Notes on Data

* The images are split into Training, Testing and Noise_Results
  * Training is the training data
  * Testing are images similar to the training data
  * Noise_Results are pictures taken outside with a diverse background

## Python Packages Used

* numpy
  * 1.17.3
* pandas
  * 0.25.3
* pytorch
  * 1.0.1
* opencv
  * 3.4.2
* scikit-image
  * 0.15.0
* scikit-learn
  * 0.21.3
* torchvision
  * 0.2.2
* matplotlib
  * 3.1.1


