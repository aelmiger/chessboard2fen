import cv2
from boardDetection import ChessboardDetector
import glob


bd = ChessboardDetector("keras_detect_model/","keras_classific_model.h5")
filenames = glob.glob("input_imgs/*")
filenames = sorted(filenames)
for file in filenames:
  img = cv2.imread(file)
  corners = bd.calcBoardCorners(img)
  bd.calcFEN(img,corners)
