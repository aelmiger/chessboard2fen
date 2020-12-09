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
    numeric_board = np.array([[11,  9,  7,  2,  4,  7,  9, 11],
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
    def predictions_to_img(self, predictions):
        """Converts predictions to FEN image format .

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        rotated_predictions = self.rotate_predictions(predictions)
        fen_string = self.predictions_to_fen(rotated_predictions)
        return self.fen2Image(fen_string)

    def predictions_to_move(self, predictions):
        """transforms a list of prediction to a chess move

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        rotated_predictions = self.rotate_predictions(predictions)
        diff = self.numeric_board - rotated_predictions
        diff_indx = np.transpose(np.nonzero(diff))
        loc_names = []
        board_changed = False
        for loc in diff_indx:
            loc_names.append(self.posChar[loc[1]]+str(8-loc[0]))
        if len(loc_names) == 2:
            move_opt_1 = chess.Move.from_uci(
                loc_names[0]+loc_names[1])
            move_opt_2 = chess.Move.from_uci(
                loc_names[1]+loc_names[0])
            if move_opt_1 in self.board.legal_moves:
                self.board.push_uci(str(move_opt_1))
                board_changed = True
            elif move_opt_2 in self.board.legal_moves:
                self.board.push_uci(str(move_opt_2))
                board_changed = True
            # else:
            #     print("No legal move in Img")
        # Casteling moves
        elif len(loc_names) == 4:
            if "a1" in loc_names and chess.Move.from_uci("e1c1") in self.board.legal_moves:
                self.board.push_uci("e1c1")
                board_changed = True
            elif "h1" in loc_names and chess.Move.from_uci("e1g1") in self.board.legal_moves:
                self.board.push_uci("e1g1")
                board_changed = True
            elif "a8" in loc_names and chess.Move.from_uci("e8c8") in self.board.legal_moves:
                self.board.push_uci("e8c8")
                board_changed = True
            elif "h8" in loc_names and chess.Move.from_uci("e8g8") in self.board.legal_moves:
                self.board.push_uci("e8g8")
                board_changed = True
        #     else:
        #         print("No legal move in Img")
        # else:
        #     print("No legal move in Img")
        if board_changed:
            self.numeric_board = rotated_predictions
            eng = self.engine.play(self.board,chess.engine.Limit(depth = 15),info = chess.engine.INFO_ALL,options={"Skill Level":0})
            info = eng.info
            self.mv = eng.move
            try:
                self.bestMv = info["pv"][0]
            except:
                pass
            try:
                self.score = (-1)**(1-info["score"].turn)*info["score"].relative.cp/100
            except:
                self.score =  "M" + str((-1)**(1-info["score"].turn)*info["score"].relative.mate())
            print(self.score)
        return board_changed

    def rotate_predictions(self, predictions):
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
                self.numeric_board-np.rot90(preds, i)))
        return np.rot90(preds, np.argmin(diffs))

    def predictions_to_fen(self, predictions):
        """Converts predictions to FEN string.

        Args:
            predictions ([type]): [description]

        Returns:
            [type]: [description]
        """
        rotated_predictions = predictions.reshape(64)
        fen_string = ""
        for i in range(8):
            empty_counter = 0
            for j in range(8):
                if self.fenChar[rotated_predictions[i*8+j]] == "e":
                    empty_counter += 1
                    if j == 7:
                        fen_string += str(empty_counter)
                else:
                    if empty_counter > 0:
                        fen_string += str(empty_counter)
                        empty_counter = 0
                        fen_string += self.fenChar[rotated_predictions[i*8+j]]
                    else:
                        fen_string += self.fenChar[rotated_predictions[i*8+j]]
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
        pil_img = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    
    def curr_board_to_img(self):
        if self.score != "M0":
            if len(self.board.move_stack) > 0:
                best_move = chess.svg.Arrow(self.bestMv.from_square,self.bestMv.to_square,color="blue")
                engine_move = chess.svg.Arrow(self.mv.from_square,self.mv.to_square,color="green")
                # svg = chess.svg.board(self.board,lastmove=self.board.peek(),arrows = [engine_move,best_move], size=350)
                svg = chess.svg.board(self.board,lastmove=self.board.peek(), size=385)
            else:
                svg = chess.svg.board(self.board, size=385)
        else:
            svg = chess.svg.board(self.board, size=385)
        png = svg2png(bytestring=svg)
        pil_img = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

