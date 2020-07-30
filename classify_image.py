# import the necessary packages
import tensorflow
print(tensorflow.__version__)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

# imagenet_utils 서브 모듈는 입력 이미지를 사전 처리하고 출력 분류를 쉽게 해독할 수 있는 편리한 기능세트
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2

print(cv2.__version__)

# 명령 행 인수
ap = argparse.ArgumentParser()
# 분류하려는 입력 이미지의 경로
ap.add_argument("-i", "--image", required=True, help="path to the input image")
# 사용하려는 사전 훈련된 CNN를 지정하는 문자열 , 기본값은 vgg16 네트워크 아키텍처
ap.add_argument("-model","--model", type=str, default="vgg16", help="name of pre-trained network to use")
args = vars(ap.parse_args())

# 명령행 인수를 통해 사전 훈련된 네트워크의 이름을 받아들이려면 모델 이름(문자열)을 실제 keras 클래스에 매핑하는 Python 사전을 정의해야합니다.

# 모델 이름을 클래스에 매핑하는 사전을 정의
# Keras 내부
# 모델 이름 문자열을 해당 클래스에 매핑하는 dictionary
MODELS = {
    "vgg16"    : VGG16,
    "vgg19"    : VGG19,
	"inception": InceptionV3,
	"xception" : Xception, # TensorFlow ONLY
	"resnet"   : ResNet50
}

# esnure a valid model name was supplied via command line argument
# 만약 --model 내부에 이름이 없다면 AssertionError 발생
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# VGG16, VGG19, and ResNet all accept 224×224 input images
# while Inception V3 and Xception require 299×299 pixel inputs

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image pre-processing function
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# Here we initialize our inputShape to be 224×224 pixels.
# We also initialize our preprocess function to be the standard preprocess_input from Keras (which performs mean subtraction).

# However, if we are using Inception or Xception,
# we need to set the inputShape to 299×299 pixels,
# followed by updating preprocess to use a separate pre-processing function that performs a different type of scaling.

# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(args["model"]))
#uses the MODELS dictionary along with the --model command line argument to grab the correct Network class.
Network = MODELS[args["model"]]
model = Network(weights="imagenet")


print("[INFO] loading and pre-processing image...")
#oads our input image from disk using the supplied inputShape to resize the width and height of the image.
image = load_img(args["image"], target_size=inputShape)
#converts the image from a PIL/Pillow instance to a NumPy array.
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

#############################
print("P:{}".format(P))
#############################

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


# 실행 명령어
# $ python classify_image.py --image images/soccer_ball.jpg --model vgg16
# $ python classify_image.py --image images/bmw.png --model vgg19
# $ python classify_image.py --image images/clint_eastwood.jpg --model resnet
# $ python classify_image.py --image images/jemma.png --model resnet
# $ python classify_image.py --image images/boat.png --model inception
# $ python classify_image.py --image images/office.png --model inception
# $ python classify_image.py --image images/scotch.png --model xception
# $ python classify_image.py --image images/tv.png --model vgg16