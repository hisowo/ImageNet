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
args = vars(ap.aprse_args())

# 명령행 인수를 통해 사전 훈련된 네트워크의 이름을 받아들이려면 모델 이름(문자열)을 실제 keras 클래스에 매핑하는 Python 사전을 정의해야합니다.

# 모델 이름을 클래스에 매핑하는 사전을 정의
# Keras 내부
# 모델 이름 문자열을 해당 클래스에 매핑하는 dictionary
MODELS = {
    "vgg16":VGG16,
    "vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}
생
# esnure 유효한 모델 이름이 명령 행 인수를 통해 제공되었습니다
# 만약 --model 내부에 이름이 없다면 AssertionError 발생
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")
