from io import BytesIO
import os
from flask import Flask, render_template, request
import base64, json, imageio, skimage
import numpy as np
from PIL import Image
from datetime import datetime
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS
from playsound import playsound

app = Flask(__name__)

conv = load_model("./models/conv_95.5.h5")
FRUITS = {0: "Apple", 1: "Banana", 2: "Grapes", 3: "Pineapple", 4:"Eye", 5:"Face", 6:"Star", 7:"Bowtie", 8:"House", 9:"Cloud",}

# return Index page
@app.route('/')
def index():
    
    return render_template('index.html')

# recognition of Doodle
@app.route('/ImageRecognition', methods=['GET', 'POST'])
def ExecPy():
    retJson = {"predict_digit" :"Err", "detect_img" :"", "centering_img" :"", "prob" :{}}
    if request.method == 'POST':
        # request.body
        postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        postImg = Image.open(postImg)
        
        postImg.save("./doodle.png")
        path = 'doodle.png'
        img = image.load_img(path,target_size=(28, 28))
        
        img = np.array(img)
        img = img[:,:,0]
        x = image.img_to_array(img)
       
        
        model = conv
        img = np.reshape(x, (28, 28, 1))
        
        # invert the colors
        #x = np.invert(img)
        for i in range(len(x)):
             for j in range(len(x)):
                 x[i][j] = 255 - x[i][j]
         # brighten the image by 60%
        for i in range(len(x)):
             for j in range(len(x)):
                 if x[i][j] > 50:
                     x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = np.interp(x, [0, 255], [-1, 1])
        val = model.predict(np.array([x]))
        pred = FRUITS[np.argmax(val)]
        
        #producing voice output
        mytext = 'this is '+pred
        ob = gTTS(text=mytext, lang='en')
        ob.save('sound.mp3')
        playsound('sound.mp3')
        
        retJson["predict_digit"]=pred
      
    return json.dumps(retJson)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')
    


