from numpy.core.defchararray import index
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import imutils
from PIL import Image
from cairosvg import svg2png
import chess
import chess.svg
from io import BytesIO

class ChessboardDetector:
    """
    Class for corner detection of a chessboard with pose estimation
    """

    camM = np.array([[3.13479737e+03, 0.00000000e+00, 2.04366415e+03],
                     [0.00000000e+00, 3.13292625e+03, 1.50698424e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distM = np.array([[2.08959569e-01, -9.49127601e-01, -
                       2.70203242e-03, -1.20066339e-04, 1.33323676e+00]])
    labelNames = ["Bauer_s", "Bauer_w", "Dame_s", "Dame_w", "Koenig_s", "Koenig_w",
                  "LEER", "Laeufer_s", "Laeufer_w", "Pferd_s", "Pferd_w", "Turm_s", "Turm_w"]
    fenChar = ["p", "P", "q", "Q", "k", "K", "e", "b", "B", "n", "N", "r", "R"]
    maxFig = [8, 8, 1, 1, 1, 1, np.inf, 2, 2, 2, 2, 2, 2]

    def __init__(self, detectModelPath, classificModelPath):
        """

        Args:
            detectModelPath (string): Folder path to Keras Pose Model
            classificModelPath (string): Folder path to Keras Classification Model
        """
        # Prevent CuDNN Error
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(
            physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Import model
        self.detectionModel = keras.models.load_model(detectModelPath)
        self.classficModel = keras.models.load_model(classificModelPath)
        self.handDetection = cv2.dnn.readNetFromDarknet("yolo_hand_model/handDetection.cfg", "yolo_hand_model/handDetection.weights")
        self.handDetection.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.handDetection.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    def calcBoardCorners(self, img):
        """ Function detects for coners of chessboard

        Args:
            img ([type]): Image in BGR order
        """
        assert img.shape[2] == 3, "image should be in color"

        self.preprocessImg(img)

        predictions = self.detectionModel.predict(
            np.expand_dims(self.img_rgb, axis=0))

        # if two points are too close -> rotate image and predict again
        if(self.overlappingPoints(predictions)):
            predictions = self.rotAndPredict(30)

        if predictions is None:
            return None
            
        self.cornerPts = self.refinePredictions(predictions)

        ih, iw = self.img_rgb.shape[:2]
        ln = self.handDetection.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.handDetection.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(self.img_rgb, 1 / 255.0, (416, 416), swapRB=False, crop=False)
        self.handDetection.setInput(blob)
        layerOutputs = self.handDetection.forward(ln)
        
        boxes = []
        confidences = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                confidence = confidences[i]
                ppCheck1 = cv2.pointPolygonTest(np.array(self.cornerPts).reshape(-1,1,2), (x, y), False)
                ppCheck2 = cv2.pointPolygonTest(np.array(self.cornerPts).reshape(-1,1,2), (x+w, y), False)
                ppCheck3 = cv2.pointPolygonTest(np.array(self.cornerPts).reshape(-1,1,2), (x, y+h), False)
                ppCheck4 = cv2.pointPolygonTest(np.array(self.cornerPts).reshape(-1,1,2), (x+w, y+h), False)
                if ppCheck1 >0 or ppCheck2 >0 or ppCheck3 >0 or ppCheck4 >0:
                    print("Hand over board")
        # cv2.imshow('img', cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        return self.cornerPts

    def calcFEN(self, img, corners):
        self.img_nn = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_nn = self.img_nn / 255

        scaleFacCornerEstim = img.shape[1]/512
        scaledCorners = np.array(corners) * scaleFacCornerEstim
        destCoords = np.array(
            [[0, 80, 0], [80, 80, 0], [80, 0, 0], [0, 0, 0]], dtype=np.float32)
        _, r, t = cv2.solvePnP(destCoords, scaledCorners,
                               self.camM, self.distM)
        imgs = []
        for i in range(8):
            for j in range(8):
                cellImg = self.getImgOfCell(7-i, j, r, t)
                # cv2.imshow("",cellImg)
                # cv2.waitKey(0)
                imgs.append(cellImg)
        imgs = np.array(imgs)
        confidence = self.classficModel.predict(imgs, batch_size=4)
        predictions = self.filterPredicted(confidence)
        fen_string = self.predictions2FEN(predictions)
        boardImg = self.fen2Image(fen_string)

        for c in self.cornerPts:
            cv2.drawMarker(img, tuple((c*scaleFacCornerEstim).astype(np.int)),
                           (0, 0, 255), markerSize=100, thickness=10)
        cv2.imshow("Image", imutils.resize(img, width=800))
        cv2.imshow("Detected Board", boardImg)
        cv2.waitKey(0)

    def fen2Image(self, fen_string):
        board = chess.Board(fen_string + " w - - 0 1")
        svg = chess.svg.board(board, size=350)
        png = svg2png(bytestring=svg)
        pilImg = Image.open(BytesIO(png)).convert('RGBA')
        return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGBA2BGRA)

    def predictions2FEN(self, predictions):
        fen_string = ""
        for i in range(8):
            empty_counter = 0
            for j in range(8):
                if self.fenChar[predictions[i*8+j]] == "e":
                    empty_counter += 1
                    if j == 7:
                        fen_string += str(empty_counter)
                else:
                    if empty_counter > 0:
                        fen_string += str(empty_counter)
                        empty_counter = 0
                        fen_string += self.fenChar[predictions[i*8+j]]
                    else:
                        fen_string += self.fenChar[predictions[i*8+j]]
            if i != 7:
                fen_string += "/"
        return fen_string

    def filterPredicted(self, confidence):
        boardLayout = -np.ones(64)
        sortedConfid = np.argsort(-np.amax(confidence, axis=-1))
        for i in range(64):
            indx = sortedConfid[i]
            unique, counts = np.unique(boardLayout, return_counts=True)
            figDict = dict(zip(unique, counts))

            tooManyOfType = True
            indxCounter = 0
            currentFig = np.argsort(-confidence[indx])
            while(tooManyOfType):
                try:
                    tooManyOfType = figDict[currentFig[indxCounter]
                                            ] >= self.maxFig[currentFig[indxCounter]]
                except KeyError:
                    tooManyOfType = False
                if (tooManyOfType):
                    indxCounter += 1
            boardLayout[indx] = currentFig[indxCounter]
        return boardLayout.astype(np.int)

    def getImgOfCell(self, cellX, cellY, r, t):
        black = np.zeros((100, 100, 1), dtype="uint8")
        cv2.circle(black, (5+10*cellX, 5+10*cellY), 4, 255, 0)
        lowPos = np.argwhere(black)
        upPos = lowPos.copy()
        upPos[:, 2] = 13
        pos = np.concatenate((lowPos, upPos)).astype(np.float32)
        imgpts, _ = cv2.projectPoints(pos, r, t, self.camM, self.distM)
        imgpts = imgpts.reshape(-1, 2)
        rect = cv2.minAreaRect(imgpts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        tempImg = cv2.warpPerspective(
            self.img_nn, M, (int(width), int(height)))
        if tempImg.shape[0] < tempImg.shape[1]:
            tempImg = cv2.rotate(tempImg, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height = tempImg.shape[0]
        width = tempImg.shape[1]
        aspRatio = width / height
        given_Ration = 1 / 2
        if aspRatio > given_Ration:
            newWidth = height * given_Ration
            diff = round((width-newWidth)/2)
            newMat = tempImg[:, diff:(width - diff), :]
        elif aspRatio < given_Ration:
            newHeight = width / given_Ration
            diff = round((height-newHeight)/2)
            newMat = tempImg[diff:(height - diff), :, :]
        else:
            newMat = tempImg
        newMat = imutils.resize(newMat, width=100)
        return cv2.resize(newMat, (100, 200)).reshape(200, 100, 3)

    def refinePredictions(self, predictions):
        """Correct estimated corner prediction by applying hough lines to a small section of image

        Args:
            predictions ([type]): predictions of nn model

        Returns:
            [type]: corrected corner points
        """
        points = predictions[0, :, 0:2].astype(int)
        # edges = self.calcImgEdges()
        edgeLines = []
        for i in range(4):
            # croppedEdges = self.cropEdgePoints(edges, points, i)
            # lines = cv2.HoughLines(croppedEdges, 1, np.pi/180, 500)
            # if lines is not None:
            #     for rho, theta in lines[0]:
            #         if theta == 0:
            #             theta += 0.01
            #         x0 = rho/np.sin(theta)
            #         x1 = -np.cos(theta)/np.sin(theta)
            #         edgeLines.append(np.array([x0, x1]))
            # else:
            x1 = (points[(i+1) % 4][1]-points[i][1]) / \
                (points[(i+1) % 4][0]-points[i][0])
            x0 = points[(i+1) % 4][1] - x1 * points[(i+1) % 4][0]
            edgeLines.append(np.array([x0, x1]))
        cornerPts = self.intersectionPts(edgeLines)
        hull = cv2.convexHull(np.array(cornerPts), clockwise=False)[:, 0]
        s = hull.sum(axis=1)[0:4]
        cornerPts = list(np.roll(hull, -np.argmin(s), axis=0))
        return cornerPts

    def calcImgEdges(self):
        """Canny edge detection

        Returns:
            [type]: edge img
        """
        gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.blur(gray, (3, 3))
        return cv2.Canny(gray, 20, 250)

    def cropEdgePoints(self, edges, points, lineIdx):
        """crop the edge image so it only includes points around the estimated line

        Args:
            edges ([type]): edge image
            points ([type]): estimated corner points
            lineIdx ([type]): index of current line

        Returns:
            [type]: image of cropped edges
        """
        black_img = np.zeros_like(edges)
        cv2.line(black_img, tuple(points[lineIdx]), tuple(
            points[(lineIdx+1) % 4]), 255, 3)
        return cv2.bitwise_and(edges, black_img, mask=black_img)

    def intersectionPts(self, lines):
        """calculate the intersection points of the detected lines

        Args:
            lines ([type]): line parameters

        Returns:
            [type]: refined corner points
        """
        intersecPts = []
        for i in range(4):
            x0 = lines[i][0]
            x1 = lines[i][1]
            x2 = lines[(i-1) % 4][0]
            x3 = lines[(i-1) % 4][1]
            cx = (x0-x2)/(x3-x1)
            cy = x3*cx+x2
            c = np.round(np.array([cx, cy])).astype(np.int)
            # p1 = np.round(np.array([0, x0])).astype(np.int)
            # p2 = np.round(np.array([img_width, x1*img_width+x0])).astype(np.int)
            # cv2.line(self.img_rgb, tuple(p1), tuple(p2), (0,0,255), 1)
            # cv2.circle(self.img_rgb,tuple(points[i]),3,255,-1)
            cv2.drawMarker(self.img_rgb, tuple(c), (0, 255, 0))
            intersecPts.append(c)
        return intersecPts

    def rotAndPredict(self, angle):
        """Rotates input img and runs prediction again

        Args:
            angle ([type]): angle in degrees

        Returns:
            [type]: returns prediction with corrected rotation
        """
        rotImg = imutils.rotate(self.img_rgb, angle=-angle)
        predictions = self.detectionModel.predict(
            np.expand_dims(rotImg, axis=0))
        if self.overlappingPoints(predictions):
            print("Could not detect 4 Corners")
            return None
        # Correct rotation of predicted points
        M = cv2.getRotationMatrix2D((256, 192), angle, 1)
        predictions[0, :, 2] = 1
        predictions[0, :, 0:2] = (M@predictions[0].T).T
        return predictions

    def overlappingPoints(self, predictions):
        """Check if points are too close to another

        Args:
            predictions ([type]): output of the nn estimation

        Returns:
            [bool]: Returns true if points are overlapping 
        """

        points = predictions[0, 0:4, 0:2]
        distances = np.triu(cdist(points, points))
        distances[distances == 0] = np.inf
        indxs = np.argwhere(distances < 30)
        return not len(indxs) == 0

    def preprocessImg(self, img):
        self.img_rgb = cv2.resize(img, (512, 384))
        self.img_rgb = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2RGB)
