from PIL import Image
from cairosvg import svg2png
import chess
import chess.svg
import chess.engine
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
    mv = None
    score = 0
    def __init__(self):
        """Initialize the chessboard
        """
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("./engine/stockfish")
    def predictions2Image(self, predictions):
        """Converts predictions to FEN image format .

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        rotPredictions = self.rotatePredictions(predictions)
        fen_string = self.predictions2FEN(rotPredictions)
        return self.fen2Image(fen_string)

    def predictions2move(self, predictions):
        """transforms a list of prediction to a chess move

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
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
                self.board.push_uci(str(moveOption1))
                boardChanged = True
            elif moveOption2 in self.board.legal_moves:
                self.board.push_uci(str(moveOption2))
                boardChanged = True
            # else:
            #     print("No legal move in Img")
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
        #     else:
        #         print("No legal move in Img")
        # else:
        #     print("No legal move in Img")
        if boardChanged:
            self.numericBoard = rotPredictions
            eng = self.engine.play(self.board,chess.engine.Limit(depth = 18),info = chess.engine.INFO_ALL,options={"Skill Level":0})
            info = eng.info
            self.mv = eng.move
            self.bestMv = info["pv"][0]
            self.score = (-1)**(1-info["score"].turn)*info["score"].relative.cp/100
            print(self.score)
        return boardChanged

    def rotatePredictions(self, predictions):
        """Rotate predictions to have white at the bottom

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        preds = predictions.reshape(8, 8)
        diffs = []
        for i in range(4):
            diffs.append(np.count_nonzero(
                self.numericBoard-np.rot90(preds, i)))
        return np.rot90(preds, np.argmin(diffs))

    def predictions2FEN(self, predictions):
        """Converts predictions to FEN string.

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        rotPredictions = predictions.reshape(64)
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
        """Converts FEN string to opencv mat.

        Args:
            fen_string ([type]): [description]

        Returns:
            [type]: [description]
        """
        board = chess.Board(fen_string + " w - - 0 1")
        svg = chess.svg.board(board, size=350)
        png = svg2png(bytestring=svg)
        pilImg = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGBA2BGRA)
    
    def currentBoard2Image(self):
        if len(self.board.move_stack) > 0:
            bestMvArr = chess.svg.Arrow(self.bestMv.from_square,self.bestMv.to_square,color="blue")
            mvArr = chess.svg.Arrow(self.mv.from_square,self.mv.to_square,color="green")
            svg = chess.svg.board(self.board,lastmove=self.board.peek(),arrows = [mvArr,bestMvArr], size=350)
        else:
            svg = chess.svg.board(self.board, size=350)
        png = svg2png(bytestring=svg)
        pilImg = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGBA2BGRA)

