from chessboard import Chessboard
import cv2
from boardDetection import ChessboardDetector
from chessboard import Chessboard
import glob

detect = ChessboardDetector("models/detection","models/classification.h5")
board = Chessboard()

def visualize():
  boardImg = board.currentBoard2Image()

  #draw board outline
  for i in range(4):
    cv2.line(detect.img_rgb,tuple(corners[i]),tuple(corners[(i+1)%4]),255,1,cv2.LINE_AA)
  #draw score
  if board.score >0:
    cv2.rectangle(detect.img_rgb,(256,10),(256-min(int(156/12*board.score),150),30),(255,255,255),-1)
  else:
    cv2.rectangle(detect.img_rgb,(256,10),(256-max(int(156/12*board.score),-150),30),(0,0,0),-1)
  cv2.rectangle(detect.img_rgb,(100,0),(412,40),(200,200,200),2)
  cv2.line(detect.img_rgb,(256,0),(256,40),(200,200,200),3)
  cv2.putText(detect.img_rgb,str(board.score),(225,67), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  #draw imgs
  cv2.imshow("Input Img", cv2.cvtColor(detect.img_rgb, cv2.COLOR_BGR2RGB))
  cv2.imshow("Board", boardImg)
  cv2.waitKey(0)

filenames = glob.glob("input_imgs/*")
filenames = sorted(filenames)
for file in filenames:
  img = cv2.imread(file)
  corners = detect.predictBoardCorners(img)
  if corners is not None:
    predictions = detect.predictBoard(img,corners)
    if board.predictions2move(predictions):
      visualize()      
