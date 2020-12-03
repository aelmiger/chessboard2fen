from chessboard import Chessboard
import cv2
from boardDetection import ChessboardDetector
from chessboard import Chessboard
import glob


detect = ChessboardDetector("keras_detect_model/","keras_classific_model.h5")
board = Chessboard()

filenames = glob.glob("input_imgs/*")
filenames = sorted(filenames)
for file in filenames:
  img = cv2.imread(file)
  corners = detect.predictBoardCorners(img)
  if corners is not None:
    predictions = detect.predictBoard(img,corners)
    boardImg = board.predictions2Image(predictions)
    for i in range(4):
      cv2.line(detect.img_rgb,tuple(corners[i]),tuple(corners[(i+1)%4]),255,2,cv2.LINE_AA)
    cv2.imshow("Input Image", cv2.cvtColor(detect.img_rgb, cv2.COLOR_BGR2RGB))
    cv2.imshow("Board", boardImg)
    cv2.waitKey(0)
