from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
class DataLoader(Dataset):
    def __init__(self, dataSet, poses, dType="train", camera=0):
        '''

        :param dataSet:
        :param poses:
        :param dType:
        :param camera:
        '''
        super(DataLoader, self).__init__()
        self.dataSet = dataSet
        self.poses = poses
        self.dType = dType
        self.camera = camera
        self.data = self.processData()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''

        :param index:
        :return:
        '''
        if torch.is_tensor(index):
            index = index.tolist()

        (left_img, right_img) = list(self.data)[index]

        left_Img = cv2.imread(left_img, flags=cv2.IMREAD_GRAYSCALE)
        right_Img = cv2.imread(right_img, flags=cv2.IMREAD_GRAYSCALE)
        f_mat = self.data[(left_img, right_img)]

        size = 64

        hl, wl = left_Img.shape[0], left_Img.shape[1]
        left_Img = left_Img[int(hl/2)-int(size/2): int(hl/2) + int(size/2), int(wl/2) -int(size/2) : int(wl/2)+int(size/2) ]

        hr, wr = right_Img.shape[0], right_Img.shape[1]

        right_Img = right_Img[int(hr / 2) - int(size/2): int(hr / 2) + int(size/2), int(wr / 2) - int(size/2): int(wr / 2) + int(size/2)]


        left_Img = (left_Img - 127.5)/127.5
        right_Img = (right_Img - 127.5)/127.5

        left_Img = np.expand_dims(left_Img, axis=2)
        right_Img = np.expand_dims(right_Img, axis=2)
        input = np.concatenate((left_Img, right_Img), axis=2)
        input = np.rollaxis(input,2,0)
        f_mat /= f_mat[8]
        return input, f_mat, (left_img, right_img)

    # reading poses at datasetID
    @staticmethod
    def transMatFrom(arr):
        '''

        :param arr:
        :return:
        '''
        if len(arr) == 12:
            result = np.eye(4)
            arr = np.array(arr).reshape((3,4))
            result[:3,:4] = arr
            return result

        return np.eye(4)

    def FMat(self, R, T):
        '''

        :param R:
        :param T:
        :return:
        '''
        cameraP = "P{}".format(self.camera)
        if self.d:
            K = self.d[cameraP][0:3, 0:3]
            t = T

            T = np.array([
                [0, -t[2], t[1]],
                [t[2], 0, -t[0]],
                [-t[1], t[0], 0]
            ], dtype=float)

            E = np.dot(R, T)
            return E.flatten()

        return np.zeros((9,1))

    def processData(self):
        '''

        :return:
        '''
        D = {}
        if self.dType == 'train' or self.dType == 'validate':
            for sequence in self.dataSet:
                data = self.dataSet[sequence]

                self.d = data['calib.txt']

                imgC = "image_{}".format(self.camera)
                images = data[imgC]

                poseC = "{}.txt".format(sequence)
                poses = self.poses[poseC]
                l = len(images)
                for i in range(l-1):
                    index1 = i
                    index2 = i+1

                    left_img = images[index1]
                    right_img = images[index2]

                    # R, T = self.convertP(self.transMatFrom(poses[index1]), self.transMatFrom(poses[index2]))
                    pose1 = self.transMatFrom(poses[index1])
                    pose2 = np.linalg.inv(pose1).dot(self.transMatFrom(poses[index2]))
                    R = pose2[:3,:3]
                    T = pose2[:3,3].flatten()
                    f_mat = self.FMat(R=R, T=T)
                    D[(left_img, right_img)] = f_mat
        else:
            for sequence in self.dataSet:
                data = self.dataSet[sequence]

                self.d = data['calib.txt']

                imgC = "image_{}".format(self.camera)
                images = data[imgC]

                l = len(images)
                for i in range(l - 1):
                    index1 = i
                    index2 = i + 1

                    left_img = images[index1]
                    right_img = images[index2]

                    D[(left_img, right_img)] = self.d
        return D




