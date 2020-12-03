from PIL import Image
from cairosvg import svg2png
import chess
import chess.svg
from io import BytesIO
import numpy as np
import cv2


class Chessboard:

    posChar = ["a", "b", "c", "d", "e", "f", "g", "h"]
    fenChar = ["p", "P", "q", "Q", "k", "K", "e", "b", "B", "n", "N", "r", "R"]
    numericBoard = np.array([[11,  9,  7,  2,  4,  7,  9, 11],
                             [0,  0,  0,  0,  0,  0,  0,  0],
                             [6,  6,  6,  6,  6,  6,  6,  6],
                             [6,  6,  6,  6,  6,  6,  6,  6],
                             [6,  6,  6,  6,  6,  6,  6,  6],
                             [6,  6,  6,  6,  6,  6,  6,  6],
                             [1,  1,  1,  1,  1,  1,  1,  1],
                             [12, 10,  8,  3,  5,  8, 10, 12]])

    def __init__(self):
        self.board = chess.Board()

    def predictions2Image(self, predictions):
        rotPredictions = self.rotatePredictions(predictions)
        fen_string = self.predictions2FEN(rotPredictions)
        return self.fen2Image(fen_string)

    def predictions2move(self, predictions):
        rotPredictions = self.rotatePredictions(predictions)
        diff = self.numericBoard - rotPredictions
        diffLocations = np.transpose(np.nonzero(diff))
        locationNames = []
        boardChanged = False
        for loc in diffLocations:
            locationNames.append(self.posChar[loc[1]]+str(8-loc[0]))
        if len(locationNames) == 2:
            moveOption1 = chess.Move.from_uci(
                locationNames[0]+locationNames[1])
            moveOption2 = chess.Move.from_uci(
                locationNames[1]+locationNames[0])
            if moveOption1 in self.board.legal_moves:
                self.board.push_uci(moveOption1)
                boardChanged = True
            elif moveOption2 in self.board.legal_moves:
                self.board.push_uci(moveOption2)
                boardChanged = True
            else:
                print("No legal move in Img")
        # Casteling moves
        elif len(locationNames) == 4:
            if "a1" in locationNames and chess.Move.from_uci("e1c1") in self.board.legal_moves:
                self.board.push_uci("e1c1")
                boardChanged = True
            elif "h1" in locationNames and chess.Move.from_uci("e1g1") in self.board.legal_moves:
                self.board.push_uci("e1g1")
                boardChanged = True
            elif "a8" in locationNames and chess.Move.from_uci("e8c8") in self.board.legal_moves:
                self.board.push_uci("e8c8")
                boardChanged = True
            elif "h8" in locationNames and chess.Move.from_uci("e8g8") in self.board.legal_moves:
                self.board.push_uci("e8g8")
                boardChanged = True
            else:
                print("No legal move in Img")
        else:
            print("No legal move in Img")
        if boardChanged:
            self.numericBoard = rotPredictions
        return boardChanged

    def rotatePredictions(self,predictions):
        preds = predictions.reshape(8,8)
        diffs = []
        for i in range(4):
            diffs.append(np.count_nonzero(self.numericBoard-np.rot90(preds,i)))
        return np.rot90(preds,np.argmin(diffs))

    def predictions2FEN(self, predictions):
        rotPredictions=self.rotatePredictions(predictions).reshape(64)
        fen_string = ""
        for i in range(8):
            empty_counter = 0
            for j in range(8):
                if self.fenChar[rotPredictions[i*8+j]] == "e":
                    empty_counter += 1
                    if j == 7:
                        fen_string += str(empty_counter)
                else:
                    if empty_counter > 0:
                        fen_string += str(empty_counter)
                        empty_counter = 0
                        fen_string += self.fenChar[rotPredictions[i*8+j]]
                    else:
                        fen_string += self.fenChar[rotPredictions[i*8+j]]
            if i != 7:
                fen_string += "/"
        return fen_string

    def fen2Image(self, fen_string):
        board = chess.Board(fen_string + " w - - 0 1")
        svg = chess.svg.board(board, size=350)
        png = svg2png(bytestring=svg)
        pilImg = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGBA2BGRA)
