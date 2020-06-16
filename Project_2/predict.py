import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from PIL import Image
import json

parser = argparse.ArgumentParser(description='Flower type predictor.')
parser.add_argument('IMG_PATH', help='Path to example image')
parser.add_argument('MODEL', help='Path to saved keras model')
parser.add_argument('--top_k', type=int, help='Return the top K most likely classes. Default value = 5')
parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names')
args = parser.parse_args()

reloaded_keras_model = tf.keras.models.load_model(args.MODEL, custom_objects={'KerasLayer':hub.KerasLayer})

top_k = 5

def process_image(image):
    tensor_image = tf.convert_to_tensor(image)
    tensor_image = tf.image.resize(tensor_image, [224,224])
    tensor_image /= 255
    return tensor_image.numpy()

def predict(image_path, model, top_k):
    np.set_printoptions(suppress=True)
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_img = process_image(test_image)
    processed_img = np.expand_dims(processed_img, axis=0)
    ps = model.predict(processed_img)
    probs, labels = tf.math.top_k(ps, k=top_k)
    return probs.numpy(), (labels+1).numpy()

if args.top_k:
    probs, labels = predict(args.IMG_PATH, reloaded_keras_model, args.top_k)
else:
    probs, labels = predict(args.IMG_PATH, reloaded_keras_model, top_k)

if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    top_labels = []
    for label in labels[0]:
        top_labels.append(class_names.get(str(label)))
    print("Top K probs = ", probs)
    print("\nTop K labels = ", top_labels)
else:
    print("Top K probs = ", probs)
    print("\nTop K labels = ", labels)
