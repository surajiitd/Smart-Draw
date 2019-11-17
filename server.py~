from io import BytesIO
from flask import Flask, render_template, request
#from ConvNeural import ConvNeuralNet
from scipy.misc import imsave, imread, imresize
import base64, json
import numpy as np
from PIL import Image
from datetime import datetime
from keras.models import load_model
from prepare_data import normalize

app = Flask(__name__)
#cnn = ConvNeuralNet()
#cnn.loadmodel()
conv = load_model("./models/conv_95.5.h5")
conv._make_predict_function()
FRUITS = {0: "Apple", 1: "Banana", 2: "Grapes", 3: "Pineapple", 4:"Eye", 5:"Face", 6:"Star", 7:"Bowtie", 8:"House", 9:"Cloud",}
# return Index page
@app.route('/')
def index():
    
    return render_template('index.html')

# recognition digit
@app.route('/DigitRecognition', methods=['GET', 'POST'])
def ExecPy():
    retJson = {"predict_digit" :"Err", "detect_img" :"", "centering_img" :"", "prob" :{}}
    if request.method == 'POST':
        # request.body
        postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        postImg = Image.open(postImg)
        postImg.save("./temp.png")
        #with open('temp.png', 'wb') as output:
            #output.write(img)
        x = imread('temp.png', mode='L')
        # resize input image to 28x28
        x = imresize(x, (28, 28))
		  # brighten the image a bit (by 60%)
        #for i in range(len(x)):
            #if x[i] > 50:
                #x[i] = min(255, x[i] + x[i] * 0.60)

        
        model = conv
        x = np.expand_dims(x, axis=0)
        x = np.reshape(x, (28, 28, 1))
         # invert the colors
        x = np.invert(x)
         # brighten the image by 60%
        for i in range(len(x)):
             for j in range(len(x)):
                 if x[i][j] > 50:
                     x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        pred = FRUITS[np.argmax(val)]
        retJson["predict_digit"]=pred
        
        # Import the required module for text to speech conversion 
        from gtts import gTTS 

        # This module is imported so that we can 
        # play the converted audio 
        import os 

        # The text that you want to convert to audio 
        mytext = 'It is a '+pred;

        # Language in which you want to convert 
        language = 'en'

        # Passing the text and language to the engine, 
        # here we have marked slow=False. Which tells 
        # the module that the converted audio should 
        # have a high speed 
        myobj = gTTS(text=mytext, lang=language) 

        # Saving the converted audio in a mp3 file named 
        # welcome 
        myobj.save("welcome.mp3") 

        # Playing the converted file 
        os.system("mpg321 welcome.mp3") 
            
            

    return json.dumps(retJson)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')
    

