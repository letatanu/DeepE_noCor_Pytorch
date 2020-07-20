import os
import numpy as np
import cv2
from ulti import traj_transform as ttf

colors = [
        (255, 102, 102),
        (102, 255, 255),
        (125, 125, 125),
        (204, 229, 255),
        (0, 0, 204)
    ]
class EpipoLine:
    def __init__(self, leftImg, rightImg, R, T):
        self.leftImg = leftImg
        self.rightImg = rightImg
        self.R = R
        self.T = T

    @staticmethod
    def epipoline(x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)

    # @staticmethod
    # def convertP(pose1, pose2):
    #     R1, T1 = pose1
    #     R2, T2 = pose2
    #     # return R2, T2
    #     newR = np.dot(np.linalg.inv(R2), R1)
    #     newT = np.dot(np.dot(np.linalg.inv(R1), R2), T2) - T1
    #     #
    #     # newR = np.dot(R2, np.linalg.inv(R1))
    #     # newT = np.dot(R1, T1-T2)
    #     return newR, newT

    def FMat(self, R, T):
        # print(T)
        t = T
        T = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], dtype=float)


        E = T.dot(R)
        # return np.dot(np.linalg.inv(K0.T), np.dot(E, K0))
        return E
        #
        # A = np.dot(np.linalg.inv(K.T), E)
        # B = np.linalg.inv(K)
        # return np.dot(A, B)

    def visualize(self, sqResultDir, img_idx, THRESHOLD=0.15):
        sift = cv2.xfeatures2d.SIFT_create()
        bf = cv2.BFMatcher()

        f_mat = self.FMat(R=self.R, T=self.T)

        left_img = cv2.imread(self.leftImg)
        left_imgG = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        left_img_line = left_img.copy()

        right_img = cv2.imread(self.rightImg)
        right_imgG = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)
        right_img_line = right_img.copy()

        (kps_left, descs_left) = sift.detectAndCompute(left_imgG, None)
        (kps_right, descs_right) = sift.detectAndCompute(right_imgG, None)

        matches = bf.knnMatch(descs_left, descs_right, k=2)
        good = []
        for m, n in matches:
            if m.distance < THRESHOLD * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(left_imgG, kps_left, right_imgG, kps_right, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # drawing epipolar line
        err_l = []
        err_r = []
        img_W = left_img.shape[1] - 1
        for color_idx, g in enumerate(good):
            # get the ids of matching feature points
            id_l, id_r = g[0].queryIdx, g[0].trainIdx

            # get the feature points in both left and right images
            x_l, y_l = kps_left[id_l].pt
            x_r, y_r = kps_right[id_r].pt

            '''Color for line'''
            color = colors[color_idx % len(colors)]

            '''Epi line on the left image'''
            # epi line of right points on the left image
            point_r = np.array([x_r, y_r, 1])

            line_l = np.dot(f_mat.T, point_r)
            # verifying points
            _, err_L = self.verify_xfx(point_r, line_l)
            err_l.append(err_L)
            # calculating 2 points on the line
            y_0 = self.epipoline(0, line_l)
            y_1 = self.epipoline(img_W, line_l)
            # drawing the line and feature points on the left image
            left_img_line = cv2.circle(left_img_line, (int(x_l), int(y_l)), radius=4, color=color)
            left_img_line = cv2.line(left_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            left_img = cv2.circle(left_img, (int(x_l), int(y_l)), radius=4, color=color)

            '''Epi line on the right image'''
            # epi line of left points on the right image
            point_l = np.array([x_l, y_l, 1])
            line_r = np.dot(f_mat, point_l)

            # verifying points
            _, err_R = self.verify_xfx(point_l, line_r)
            err_r.append(err_R)
            # calculating 2 points on the line
            y_0 = self.epipoline(0, line_r)
            y_1 = self.epipoline(img_W, line_r)

            print("Point {}: ".format(color_idx), self.verify_xFx(point_l, f_mat, point_r))

            # drawing the line on the right image
            right_img_line = cv2.circle(right_img_line, (int(x_r), int(y_r)), radius=4, color=color)
            right_img_line = cv2.line(right_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            right_img = cv2.circle(right_img, (int(x_r), int(y_r)), radius=4, color=color)

        l_avgErr = np.average(err_l) if err_l else 0
        r_avgErr = np.average(err_r) if err_r else 0

        shape = left_img.shape
        emptyImg = np.ones((20, shape[1], 3))
        # vis1 = np.concatenate((left_img, right_img_line), axis=0)
        # vis2 = np.concatenate((left_img_line, right_img), axis=0)
        # vis = np.concatenate((vis1, emptyImg, vis2), axis=0)
        vis = np.concatenate((left_img_line, right_img_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(l_avgErr), (10, 20), font, 0.5, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(r_avgErr), (10, img_H - 10), font, 0.5, color=(0, 255, 0), lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(sqResultDir, 'epipoLine_sift_{}.png'.format(img_idx)), vis)
        print(os.path.join(sqResultDir, 'epipoLine_sift_{}.png'.format(img_idx)))

    @staticmethod
    def verify_xFx(point1, F, point2):
        return point2.T.dot(F).dot(point1)


    @staticmethod
    def verify_xfx(point, l):
        threshold = 2
        l = l.flatten()
        # K = EpiLine.d['P0'][0:3, 0:3]
        result = abs(np.dot(point, l.T) / np.sqrt(l[0] * l[0] + l[1] * l[1]))

        if result <= threshold:
            # print(True, result)
            return (True, result)
        # print(False, result)
        return (False, result)

dataDir = "/media/slark/DuLieuXin/Projects/deepF_noCorrs_Pytorch/dataset/Easy"
sequences = os.listdir(dataDir)
for sq in sequences:
    sqDir = os.path.join(dataDir, sq)
    poseFile = os.path.join(sqDir, "pose_left.txt")
    imgLDir = os.path.join(sqDir, "image_left")

    assert os.path.isdir(imgLDir) and os.path.isfile(poseFile), "The imge folder and pose file do not exist."
    poses = np.loadtxt(poseFile)
    # with open(poseFile) as f:
    #     for r in f:
    #         cont = np.array(contentPF.split(" "), dtype=float)
    #         assert len(cont) == 7, "the row content in the pose file is not valid"

    Rs = []
    Ts = []
    # traj_ses = ttf.shift0(np.array(poses))
    # traj_ses = tf.pos_quats2SE_matrices(np.array(poses))
    traj_ses = ttf.cam2nedSE(np.array(poses))
    # traj_ses = tf.pos_quats2SE_matrices(traj_ses)

    imgesPath = os.listdir(imgLDir)
    imgesPath.sort()
    l = len(imgesPath)

    for i in range(l-1):
        index1 = i
        index2 = i + 1

        traj1 = traj_ses[index1]
        traj1_inv = np.linalg.inv(traj1)

        traj2 = traj1_inv.dot(traj_ses[index2])
        # traj2 = traj_ses[index2]
        # traj2 = np.linalg.inv(traj2)

        # traj2 = traj_ses[index2]
        # traj2 = traj_init_inv.dot(traj2)

        # R1 = traj1[:3,:3]
        # traj1 = tf.pos_quat2SE(traj1)
        # traj2 = tf.pos_quat2SE(traj2)

        R2 = traj2[:3,:3]

        # T1 = np.array(traj1[:3,3])
        # T1.flatten()
        # T1 = np.array([T1[2],T1[0], T1[1]])

        T2 = np.array(traj2[:3,3])
        T2.flatten()

        # newR = R2
        # newT = T2
        #
        newR = R2
        newT = T2
        # #
        # newR = R2
        # newT = T2
        # newR = np.dot(R2, np.linalg.inv(R1))
        # newT = np.dot(R1, T1-T2)
        lImg = os.path.join(imgLDir, imgesPath[index1])
        rImg = os.path.join(imgLDir, imgesPath[index2])
        a = EpipoLine(leftImg=lImg, rightImg=rImg, R=newR, T=newT)
        a.visualize(sqResultDir=sqDir,img_idx=i, THRESHOLD=0.1)

        break


