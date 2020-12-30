import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import imutils


class ChessboardDetector:
    """
    Class for corner detection of a chessboard with pose estimation
    """

    cam_m = np.array([[3.13479737e+03, 0.00000000e+00, 2.04366415e+03],
                     [0.00000000e+00, 3.13292625e+03, 1.50698424e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_m = np.array([[2.08959569e-01, -9.49127601e-01, -
                       2.70203242e-03, -1.20066339e-04, 1.33323676e+00]])

    # cam_m = np.array([[1.52065634e+03, 0.00000000e+00, 9.73593337e+02],
    #                   [0.00000000e+00, 1.52151939e+03, 5.40215278e+02],
    #                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # dist_m = np.array([[2.71779902e-01, -1.43729610e+00, -
    #                     6.41435289e-04, -4.28961056e-04, 2.29535075e+00]])

    labelNames = ["Bauer_s", "Bauer_w", "Dame_s", "Dame_w", "Koenig_s", "Koenig_w",
                  "LEER", "Laeufer_s", "Laeufer_w", "Pferd_s", "Pferd_w", "Turm_s", "Turm_w"]
    maxFig = [8, 8, 1, 1, 1, 1, np.inf, 2, 2, 2, 2, 2, 2]
    dest_coords = np.array(
        [[0, 80, 0], [80, 80, 0], [80, 0, 0], [0, 0, 0]], dtype=np.float32)
    counter = 0

    def __init__(self, detect_model_path, classification_model_path):
        """

        Args:
            detect_model_path (string): Folder path to Keras Pose Model
            classification_model_path (string): Folder path to Keras Classification Model
        """
        # Speed up inference
        tf.config.optimizer.set_jit(True)
        # Prevent CuDNN Error
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Import models
        self.detection_model = keras.models.load_model(detect_model_path)
        self.classification_model = keras.models.load_model(
            classification_model_path)

    def predict_board_corners(self, img):
        """ Function detects for coners of chessboard

        Args:
            img ([type]): Image in BGR order
        """
        assert img.shape[2] == 3, "image should be in color"

        # convert img to right size and rgb order
        self.preprocessImg(img)

        # estimate 4 board corners
        predictions = self.detection_model.predict(
            np.expand_dims(self.img_rgb, axis=0))

        # if two points are too close -> rotate image by 45degrees and predict again
        if(self.overlappingPoints(predictions)):
            predictions = self.rotate_and_predict(30)

        if predictions is None:
            cv2.putText(self.img_rgb, "Cant detect board", (225, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            return []

        # sort chessboard cornerpoints in a clockwise fashion starting with top left point
        self.corner_pts = self.refine_predictions(predictions)
        return self.corner_pts

    def predictBoard(self, img, corners):
        """Predict board with given image and corners .

        Args:
            img ([type]): [description]
            corners ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.img_nn = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_nn = self.img_nn / 255
        scale_factor = img.shape[1]/512
        scaled_corners = np.array(corners) * scale_factor
        _, r, t = cv2.solvePnP(self.dest_coords, scaled_corners,
                               self.cam_m, self.dist_m)
        imgs = []
        for i in range(8):
            for j in range(8):
                cell_img = self.get_cell_img(7-i, j, r, t)
                imgs.append(cell_img)
        imgs = np.array(imgs)
        confidence = self.classification_model.predict(imgs, batch_size=8)
        predictions = self.filter_predictions(confidence)
        # self.writeCellImgsToFolder(predictions,imgs)
        return predictions

    def filter_predictions(self, confidence):
        """Filter out predictions of chess pieces by logic (only one king on board etc.)

        Args:
            confidence ([type]): confidence of classification prediction

        Returns:
            [type]: filtered predictions
        """
        board_layout = -np.ones(64)
        sort_confidence = np.argsort(-np.amax(confidence, axis=-1))
        for i in range(64):
            indx = sort_confidence[i]
            unique, counts = np.unique(board_layout, return_counts=True)
            fig_dict = dict(zip(unique, counts))

            too_many_of_fig = True
            indx_counter = 0
            curr_fig = np.argsort(-confidence[indx])
            while(too_many_of_fig):
                try:
                    too_many_of_fig = fig_dict[curr_fig[indx_counter]
                                               ] >= self.maxFig[curr_fig[indx_counter]]
                except KeyError:
                    too_many_of_fig = False
                if (too_many_of_fig):
                    indx_counter += 1
            board_layout[indx] = curr_fig[indx_counter]
        return board_layout.astype(np.int)

    def get_cell_img(self, cellX, cellY, r, t):
        """Get the image of a given cell .

        Args:
            cellX ([type]): [description]
            cellY ([type]): [description]
            r ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """
        black = np.zeros((100, 100, 1), dtype="uint8")
        cv2.circle(black, (5+10*cellX, 5+10*cellY), 5, 255, 0)
        low_pos = np.argwhere(black)
        up_pos = low_pos.copy()
        up_pos[:, 2] = 13
        pos = np.concatenate((low_pos, up_pos)).astype(np.float32)
        img_pts, _ = cv2.projectPoints(pos, r, t, self.cam_m, self.dist_m)
        img_pts = img_pts.reshape(-1, 2)
        rect = cv2.minAreaRect(img_pts)
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
        temp_img = cv2.warpPerspective(
            self.img_nn, M, (int(width), int(height)))
        if temp_img.shape[0] < temp_img.shape[1]:
            temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height = temp_img.shape[0]
        width = temp_img.shape[1]
        aspRatio = width / height
        given_ratio = 1 / 2
        if aspRatio > given_ratio:
            new_width = height * given_ratio
            diff = round((width-new_width)/2)
            new_mat = temp_img[:, diff:(width - diff), :]
        elif aspRatio < given_ratio:
            new_height = width / given_ratio
            diff = round((height-new_height)/2)
            new_mat = temp_img[diff:(height - diff), :, :]
        else:
            new_mat = temp_img
        new_mat = imutils.resize(new_mat, width=100)
        return cv2.resize(new_mat, (100, 200)).reshape(200, 100, 3)

    def refine_predictions(self, predictions):
        """Sort predicted corners in a clockwise fashion. Top left corner is the first element

        Args:
            predictions ([type]): predictions of nn model

        Returns:
            [type]: corrected corner points
        """
        points = predictions[0, :, 0:2].astype(int)
        hull = cv2.convexHull(points, clockwise=False)[:, 0]
        s = hull.sum(axis=1)[0:4]
        corner_pts = list(np.roll(hull, -np.argmin(s), axis=0))
        return corner_pts

    def rotate_and_predict(self, angle):
        """Rotates input img and runs prediction again

        Args:
            angle ([type]): angle in degrees

        Returns:
            [type]: returns prediction with corrected rotation
        """
        rot_imgs = imutils.rotate(self.img_rgb, angle=-angle)
        predictions = self.detection_model.predict(
            np.expand_dims(rot_imgs, axis=0))
        if self.overlappingPoints(predictions):
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
        """Preprocess the image before processing it .

        Args:
            img ([type]): [description]
        """
        self.img_rgb = cv2.resize(img, (512, 384))
        self.img_rgb = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2RGB)

    def writeCellImgsToFolder(self, predictions, imgs):
        """Creating training data. Predict board and save each cell img in subfolder

        Args:
            predictions ([type]): filtered predictions
            imgs ([type]): list of cell imgs
        """
        for i in range(64):
            cv2.imwrite('labeled_Imgs/'+str(self.labelNames[predictions[i]])+"/"+str(
                self.counter)+".png", cv2.cvtColor((imgs[i]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            self.counter += 1
