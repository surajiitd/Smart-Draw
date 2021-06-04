﻿## SmartDraw WebApp by CNN using TensorFlow and Flask
A Flask WebApp for recognizing scrambled images using Convolution Neural Network. And getting the voice as well as text prediction.

![gif](./demo_snapshots/snapshots.gif)


#### Neural Network
  
- Two Convolution - MaxPooling layer
- Softmax loss 
- Adam Optimizer
- Dropout
- Flatten
- Automatic Centering the Input Digit
- 94.6% of accuracy at Google's quick_draw dataset.


#### Web Application

- Flask for the backbone (The Demo Site consists of Nginx + uWSGI)

### Requirement
- Python 3.6
- gTTS==2.0.4  (for text to speech conversion)
- TensorFlow 1.9.0
- Keras 2.2.4
- Flask
- NumPy
- PIL(pillow)

I have explicitly provided requirements.txt file for all other requirements.
to install it 
### Usage

- #### Training Model
	We are training our model, then using the trained model named "conv_95.5.h5" to predict the result.
	Google colab's link: https://colab.research.google.com/drive/1QbQJlLidgOe_bKuqE65CoYDcKA7L0klL
  ```
  - python train.py 
   ```
- #### Run WebApp
  ```
  - python3 server.py
    ->access to localhost:5000
  ```
  
