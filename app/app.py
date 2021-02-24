from pprint import pprint

from flask import Flask, request, jsonify, render_template
import base64, json
from io import BytesIO
import numpy as np
import subprocess
from PIL import Image
from scipy import misc
# declare constants
HOST = '0.0.0.0'
PORT = 8888

app = Flask(__name__)

@app.route('/') # flask에서는 장식자@ 는 URL연결에 사용딘다.
def home():
    return render_template("home.html") #render_template 를 이용해 html을 불러온다

@app.route('/predict', methods=['GET','POST'])
def predict():
    pprint(request.files['image'].save("./keeponeimage/one.png"))  # 항상 같은이름으로 저

    subprocess.run(["python", "online_demo_imgcv.py"])  # 현 폴더에 있는 사진 전부를 가져가서 추론한다. 하나만 존재하도록 관리할것

    # 결과는 이미지파일
    result = './result.png'

    # output data
    return json.dumps(result)

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)