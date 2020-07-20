import os
import numpy as np
import pickle

class DataSet:
    def __init__(self, sqPath, posePath):
        '''

        :param sqPath:
        :param posePath:
        '''
        self.sqPath = sqPath
        self.posePath = posePath
        self.loadDataSet()


    def dataSets(self, k_fold, kth):
        '''

        :param k_fold:
        :param kth:
        :return:
        '''
        l = len(self.sequenceFiles)
        indexes = ["{}".format(x).zfill(2) for x in range(l)]
        trainIdxs = indexes[:kth] + indexes[kth+k_fold:11]
        valIdxs = indexes[kth:kth+k_fold]
        testIdxs = indexes[11:]
        trainSet = {x: self.sequenceFiles[x] for x in trainIdxs}
        valSet = {x: self.sequenceFiles[x] for x in valIdxs}
        testSet = {x: self.sequenceFiles[x] for x in testIdxs}
        return trainSet, valSet, testSet


    def dumpDataSet(self):
        '''

        :return:
        '''
        self.sequenceFiles = self.readFiles()
        self.poses = self.readPoses(self.posePath)
        with open("sequenceFiles.txt", "wb") as f:
            pickle.dump(self.sequenceFiles, f)

        with open("poses.txt", "wb") as f:
            pickle.dump(self.poses, f)

    def loadDataSet(self):
        '''

        :return:
        '''
        if not os.path.isfile("sequenceFiles.txt") or not os.path.isfile("poses.txt"):
            self.dumpDataSet()
            return

        with open("sequenceFiles.txt", "rb") as f:
            self.sequenceFiles=pickle.load( f)

        with open("poses.txt", "rb") as f:
            self.poses = pickle.load(f)




    def readPoses(self, posePath):
        '''
        :param posePath:
        :return: adding poses to sequenceFiles
        '''
        assert os.path.isdir(posePath), "The pose path must exist."
        poseFiles = os.listdir(posePath)
        poseFiles.sort()
        p = {}
        for f in poseFiles:
            pFile = os.path.join(posePath, f)
            with open(pFile) as pF:
                poses = np.zeros((0, 12))
                for index, contentPF in enumerate(pF):
                    cont = np.array(contentPF.split(" "), dtype=float)
                    if len(cont) == 12:
                        cont = np.array([cont])
                        poses = np.concatenate((poses, cont), axis=0)
                p[f] = poses
        return p

    def readFiles(self):
        '''
        :return: a dict of files:
            sequence:
                img_0:
                    files
        '''
        sequenceFiles = {}

        assert os.path.isdir(self.sqPath), "The path of sequences must exist."
        sequences = os.listdir(self.sqPath)
        sequences.sort()
        for sqIndex, s in enumerate(sequences):
            sequence = os.path.join(self.sqPath, s)
            if not s in sequenceFiles:
                sequenceFiles[s] = {}
            assert sequence, "The path of sequence must exist."
            imageSetDirs = os.listdir(sequence)

            for setDirIndex, setDir in enumerate(imageSetDirs):
                setDirPath = os.path.join(sequence, setDir)

                if os.path.isfile(setDirPath):
                    if "calib" in setDir:
                        d = {}
                        with open(setDirPath) as f:
                            for l in f:
                                [k, v] = l.split(":")
                                k, v = k.strip(), v.strip()
                                v = np.array([x for x in v.split(" ")], dtype=float)
                                if len(v) == 12:
                                    v = v.reshape((3, 4))
                                d[k] = v
                        sequenceFiles[s][setDir] = d
                    else:
                        a = []
                        with open(setDirPath) as f:
                            for l in f:
                                a.append(l)
                        sequenceFiles[s][setDir] = np.array(a, dtype=float)
                else:
                    imgFiles = os.listdir(setDirPath)
                    imgFiles.sort()
                    imgs = [os.path.join(setDirPath, img) for img in imgFiles]
                    sequenceFiles[s][setDir] = np.array(imgs)
        return sequenceFiles