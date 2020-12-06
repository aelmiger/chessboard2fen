from chessboard import Chessboard
import cv2
from boardDetection import ChessboardDetector
from chessboard import Chessboard
import glob
import os
label_names = ["Bauer_s","Bauer_w","Dame_s","Dame_w","Koenig_s","Koenig_w","LEER","Laeufer_s","Laeufer_w","Pferd_s","Pferd_w","Turm_s","Turm_w"]
for label in label_names:
    if not os.path.exists('labeled_Imgs/'+label):
        os.makedirs('labeled_Imgs/'+label)

detect = ChessboardDetector("keras_detect_model/","keras_classific_model.h5")
board = Chessboard()

filenames = glob.glob("input_imgs/*")
filenames = sorted(filenames)
for file in filenames:
  img = cv2.imread(file)
  corners = detect.predictBoardCorners(img)
  if corners is not None:
    predictions = detect.predictBoard(img,corners)
    if board.predictions2move(predictions):
      # detBoardImg = board.predictions2Image(predictions)
      boardImg = board.currentBoard2Image()
      for i in range(4):
        cv2.line(detect.img_rgb,tuple(corners[i]),tuple(corners[(i+1)%4]),255,1,cv2.LINE_AA)
      if board.score >0:
        cv2.rectangle(detect.img_rgb,(256,10),(256-min(int(156/12*board.score),150),30),(255,255,255),-1)
      else:
        cv2.rectangle(detect.img_rgb,(256,10),(256-max(int(156/12*board.score),-150),30),(0,0,0),-1)
      cv2.rectangle(detect.img_rgb,(100,0),(412,40),(200,200,200),2)
      cv2.line(detect.img_rgb,(256,0),(256,40),(200,200,200),3)
      cv2.putText(detect.img_rgb,str(board.score),(225,67), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      # cv2.imshow("Input Image", cv2.resize(img,(int(4032/4),int(3024/4))))
      cv2.imshow("Input Img", cv2.cvtColor(detect.img_rgb, cv2.COLOR_BGR2RGB))
      cv2.imshow("Board", boardImg)
      # cv2.imshow("DetBoard", detBoardImg)
      cv2.waitKey(0)
