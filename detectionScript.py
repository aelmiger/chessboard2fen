from chessboard import Chessboard
import cv2
from boardDetection import ChessboardDetector
from chessboard import Chessboard
import glob
import numpy as np


class chess_digitizer:
    def visualize(self, detect, board, corners):
        board_img = board.curr_board_to_img()

        # draw board outline
        for i in range(len(corners)):
            cv2.line(detect.img_rgb, tuple(corners[i]), tuple(
                corners[(i+1) % len(corners)]), 255, 1, cv2.LINE_AA)
        # draw score
        try:
            if board.score > 0:
                cv2.rectangle(detect.img_rgb, (256, 10), (256 -
                                                        min(int(156/12*board.score), 150), 30), (255, 255, 255), -1)
            else:
                cv2.rectangle(detect.img_rgb, (256, 10), (256 -
                                                        max(int(156/12*board.score), -150), 30), (0, 0, 0), -1)
        except:
            sc = int(board.score[1:])
            if sc > 0:
                cv2.rectangle(detect.img_rgb, (256, 10), (100, 30), (255, 255, 255), -1)
            elif sc == 0:
                pass
            else:
                cv2.rectangle(detect.img_rgb, (256, 10), (412, 30), (0, 0, 0), -1)

        cv2.rectangle(detect.img_rgb, (100, 0), (412, 40), (200, 200, 200), 2)
        cv2.line(detect.img_rgb, (256, 0), (256, 40), (200, 200, 200), 3)
        cv2.putText(detect.img_rgb, str(board.score), (225, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # draw imgs
        tp_img = np.hstack((cv2.cvtColor(detect.img_rgb, cv2.COLOR_BGR2RGB),board_img[0:384,0:384,0:3]))
        cv2.imshow("Board", tp_img)
        cv2.waitKey(0)

    def main(self):
        detect = ChessboardDetector(
            "models/detection", "models/classification.h5")
        board = Chessboard()
        # cap = cv2.VideoCapture('input_imgs/20201209_111609.mp4')


        filenames = glob.glob("input_imgs/*")
        filenames = sorted(filenames)
        for file in filenames:
            img = cv2.imread(file)
            corners = detect.predict_board_corners(img)
            if len(corners) == 4:
                predictions = detect.predictBoard(img, corners)
                board.predictions_to_move(predictions)
            self.visualize(detect, board, corners)
        # while True:
        #     ret, img = cap.read()
        #     # ret, img = cap.read()
        #     img = img[:,240:1680,:]
        #     corners = detect.predict_board_corners(img)
        #     if len(corners) == 4:
        #         predictions = detect.predictBoard(img, corners)
        #         # if board.predictions_to_move(predictions):
        #         board.predictions_to_move(predictions)
        #     self.visualize(detect, board, corners)

if __name__ == "__main__":
    cd = chess_digitizer()
    cd.main()
