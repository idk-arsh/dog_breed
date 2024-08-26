from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import base64
import tensorflow_hub as hub
import tf_keras
import keras
from keras import layers
import numpy as np

model = tf_keras.models.load_model('./model.h5',
                                                  custom_objects={'KerasLayer':hub.KerasLayer})
breed_list=['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset',
 'beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound',
 'bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel',
 'bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole',
 'dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer',
 'giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound',
 'japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever',
 'lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound',
 'papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier',
 'shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle',
 'standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

IMG_SIZE = 224
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('r.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        try:
            if file and allowed_file(file.filename):
                image_data = file.read()
                image = tf.io.decode_jpeg(image_data, channels=3)
                image = tf.image.convert_image_dtype(image, tf.float32)
                image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
                image = tf.expand_dims(image, axis=0)

                pred = model.predict(image)
                pred_labels = get_preds_labels(pred, breed_list)
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                image_ext = file.filename.rsplit('.', 1)[1].lower()


                return render_template('wtf.html', prediction=pred_labels,image_data=image_b64, image_ext=image_ext)
            else:
                return render_template('r.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG image.")
        except Exception as e:
            return render_template('r.html', error=f"An error occurred: {str(e)}")
    else:
        return render_template('r.html')

def get_preds_labels(preds, blist):
    pred_labels = []
    for pred in preds:
        pred_label = blist[np.argmax(pred)]
        pred_labels.append(pred_label)
    return pred_labels

if __name__ == '__main__':
    app.run(debug=True)