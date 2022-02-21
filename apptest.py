#import sqlalchemy
from imutils.video import VideoStream
from flask import Response, Flask, render_template, request, send_from_directory, redirect, url_for
from flask_migrate import Migrate
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
from PIL import Image
import pickle
from flask_sqlalchemy import SQLAlchemy
import psycopg2
import io
import base64
from flask_socketio import SocketIO, emit
import sys
from flask_cors import CORS, cross_origin
from camera import Camera
import boto3
import aws_config
import os.path




app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://lnisnsukhcqzop:bb96976d926974387df7c49d4ff570a73b00e1b02629cba11687444dfa15ee14@ec2-34-230-198-12.compute-1.amazonaws.com:5432/d7nct4sujlsnie'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:pwd@db:5432/my_db'

db = SQLAlchemy(app)

CORS(app)
socketio = SocketIO(app, always_connect=True, engineio_logger=False)
camera = Camera()

uName = ""
tempUName = ""
faceDetected = False
recognizer = None
labels = {}
labels_old = {}
snapflag = False
trainflag = False
trainedFlag = False
uFolder = ""
foldCount = 0

MIGRATE = Migrate(app, db)
PYTHONUNBUFFERED=True

aws_config.setVar()

s3 = boto3.client('s3')
s3.put_object(Bucket="cant-forget-your-face", Key='tmp/')

if not os.path.isdir('tmp/'):
    os.mkdir('tmp/')

with app.app_context():
    db.create_all()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

    password = db.Column(db.String(80), unique=False, nullable=False)
    text = db.Column(db.Text(), unique=False, nullable = True)
    #model = db.Column(db.Blob(), unique=False, nullable = True)

    def __init__(self, username=None, password=None, text = None):
        self.username = username
        self.password = password
        self.text = text


@socketio.on('image')
def image(data_image, username, capflag, snapflag, verifyflag):
    global faceDetected
    global foldCount, uFolder, trainflag, trainedFlag, tempUName
    tempUName = username

    input = data_image.split(",")[1]
    camera.enqueue_input(input, username, capflag, snapflag, verifyflag)




#Home Page landing
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/<path:path>')
def serve(path):
    print('Path:', file=sys.stderr)
    print(path, file=sys.stderr)
    if path == 'login.html':
        print('Serving Login Page', file=sys.stderr)
    return render_template(path)
    

def train(username):
    s3 = boto3.client('s3')
    image_dir = 'tmp/' + username  # grab image directory path

    cascPath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    recognizer = cv2.face_LBPHFaceRecognizer.create()

    current_id = 0
    label_ids = {}

    x_train = []
    y_labels = []

    for root, dir, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
                path = os.path.join(root, file)
                s3.upload_file('tmp/' + username + '/' + file, "cant-forget-your-face", 'tmp/' + username + '/' + file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()  # set label to folder name
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id  # add label to dictionary
                    current_id += 1  # value of dict
                id_ = label_ids[label]
                pil_image = Image.open(path).convert("L")
                size = (550, 550)
                res_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(res_image, "uint8")  # convert image to numpy array for training
                faces = faceCascade.detectMultiScale(image_array, 1.1, 4)
                for (x, y, w, h) in faces:
                    face = image_array[y:y + h, x:x + w]
                    x_train.append(face)  # append face to model
                    y_labels.append(id_)  # append label

    with open(image_dir + "/labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(image_dir + "/trainer.yml")
    s3.upload_file('tmp/' + username + '/trainer.yml', "cant-forget-your-face", 'tmp/' + username + '/trainer.yml')

#Register Page Action: Register
@app.route('/createAcc.html', methods = ['POST'])
def register_post():
    global trainflag, trainedFlag, uName
    db.create_all()
    db.session.commit()
    usern = request.form['username']
    pass1 = request.form['password1']
    pass2 = request.form['password2']
    if usern == "" or pass1 == "":
        return "must not be empty"
    user = User.query.filter_by(username=usern).first()
    if user is not None:
        return "account name already in use"
    if pass1 == pass2:
        newUser = User(username = usern, password = pass1, text = "Type in this box")
        db.session.add(newUser)
        db.session.commit()
    else:
        return "pass not same"
    user = User.query.filter_by(username=usern).first()
    if user is None:
        return "err: user not created"
    # if trainedFlag is False:
    #     return "err: picture not captured"
    else:
        train(user.username)
        uName = user.username
        # createAndSave()
        trainflag = True
        return render_template("TextEdit.html", text = user.text) #redirect(url_for("http://localhost:5000/textEdit.html", text = "user text"))


def gen():

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/login", methods = ['POST'])
def login():
    global faceDetected
    global uName


    db.create_all()
    db.session.commit()

    form_username = request.form['username']
    form_password = request.form['password']
    
    user = User.query.filter_by(username = form_username).first()

    if(user is None):
        return "Username doesn't exist"

    if(user.password == form_password and camera.get_access()):
        uName = user.username
        
        return render_template("TextEdit.html", text = user.text)
    else:
        return "Incorrect password or face not recognized"

@app.route("/save", methods = ['POST'])
def save():
    global uName
    user = User.query.filter_by(username = uName).first()

    if user is None:
        return "it broke"

    user.text = request.form['classic']
    db.session.commit()
    return render_template("TextEdit.html", text = User.query.filter_by(username = uName).first().text)

@app.route("/logout", methods = ['POST'])
def logout():
    global uName
    global faceDetected
    faceDetected = False
    uName = ""
    return redirect("/index.html")

@app.route("/loginFirst", methods = ['POST'])
def loginFirst():
    global uName
    if uName != "":
        user = User.query.filter_by(username = uName).first()
        return render_template("TextEdit.html", text = user.text)
    else:
        return redirect("/login.html")

    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(debug=True, host='0.0.0.0', port=port)