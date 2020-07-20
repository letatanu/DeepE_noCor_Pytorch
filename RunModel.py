from models.DataLoader import DataLoader
from models.DataSet import DataSet
from models.models import DeepFMatNet, DeepFMatAlex, DeepFMatVGG16, DeepFMatResNet18
from models.Regularizer import L2Regularizer, L1Regularizer
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import os
import cv2
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import argparse
import datetime

# torch.autograd.set_detect_anomaly(True)


def epipoline(x, formula):
    '''

    :param x:
    :param formula:
    :return:
    '''
    array = formula.flatten()
    a = array[0]
    b = array[1]
    c = array[2]
    return int((-c - a * x) / b)

def verify_xfx(line, point):
    '''

    :param line:
    :param point:
    :return:
    '''
    l = np.array(line).flatten()
    a = l[0]
    b = l[1]
    return abs(line.dot(point))/np.sqrt(a*a+b*b)

def visualize2Images(args, left_paths, right_paths, f_mats, loss, epoch, img_idx, visualizeDir = "visualization"):
    '''

    :param args:
    :param left_paths:
    :param right_paths:
    :param f_mats:
    :param loss:
    :param epoch:
    :param img_idx:
    :param visualizeDir:
    :return:
    '''
    colors = [
        (255, 102, 102),
        (102, 255, 255),
        (125, 125, 125),
        (204, 229, 255),
        (0, 0, 204)
    ]
    f_mats = f_mats.cpu().numpy()
    THRESHOLD = 0.12
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    images = []
    errors = {}
    for idx, (left_path, right_path, f_mat) in enumerate(zip(left_paths, right_paths, f_mats)):
        f_mat = np.array(f_mat.reshape((3,3)))

        left_img = cv2.imread(left_path)
        #-------
        hl, wl = left_img.shape[0], left_img.shape[1]
        left_img = left_img[int(hl / 2) - 128: int(hl / 2) + 128, int(wl / 2) - 128: int(wl / 2) + 128]
#--------------------------

        left_imgG = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        left_img_line = left_img.copy()

        right_img = cv2.imread(right_path)
#-------------------------------
        hr, wr = right_img.shape[0], right_img.shape[1]
        right_img = right_img[int(hr / 2) - 128: int(hr / 2) + 128, int(wr / 2) - 128: int(wr / 2) + 128]

        right_imgG = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)
        right_img_line = right_img.copy()

        (kps_left, descs_left) = sift.detectAndCompute(left_imgG, None)
        (kps_right, descs_right) = sift.detectAndCompute(right_imgG, None)

        matches = bf.knnMatch(descs_left, descs_right, k=2)
        good = []
        for m, n in matches:
            if m.distance < THRESHOLD * n.distance:
                good.append([m])

        err_l = []
        err_r = []
        img_W = left_img.shape[1] - 1
#---------------------------------------------------------------------
        for color_idx, g in enumerate(good):
            # get the ids of matching feature points
            id_l, id_r = g[0].queryIdx, g[0].trainIdx
            # x: column
            # y: row
            # get the feature points in both left and right images
            x_l, y_l = kps_left[id_l].pt
            x_r, y_r = kps_right[id_r].pt

            '''Color for line'''
            color = colors[color_idx % len(colors)]

            '''Epi line on the left image'''
            # epi line of right points on the left image
            point_r = np.array([x_r, y_r, 1])
            line_l = np.dot(f_mat.T, point_r)

          
            # calculating 2 points on the line
            y_0 = epipoline(0, line_l)
            y_1 = epipoline(img_W, line_l)
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
            err_R = verify_xfx(line_r, point_r)
            err_r.append(err_R)
            # verifying points
            err_L = verify_xfx(line_l, point_l)
            err_l.append(err_L)
            # calculating 2 points on the line
            y_0 = epipoline(0, line_r)
            y_1 = epipoline(img_W, line_r)

            # drawing the line on the right image
            right_img_line = cv2.circle(right_img_line, (int(x_r), int(y_r)), radius=4, color=color)
            right_img_line = cv2.line(right_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            right_img = cv2.circle(right_img, (int(x_r), int(y_r)), radius=4, color=color)

        l_avgErr = np.average(err_l) if err_l else 0
        r_avgErr = np.average(err_r) if err_r else 0

        vis = np.concatenate((left_img_line, right_img_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, '{:.4f}'.format(float(l_avgErr)), (10, 20), font, 0.3, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, '{:.4f}'.format(float(r_avgErr)), (10, img_H - 10), font, 0.3, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, '{:.4f}'.format(float(loss.data.cpu())), (int(img_W-img_W/2), img_H - 10), font, 0.3, color=(0, 255, 0), lineType=cv2.LINE_AA)

        sqResultDir = os.path.join(ROOT_DIR, visualizeDir, '{}'.format(epoch))
        if not os.path.exists(sqResultDir):
            os.makedirs(sqResultDir)

        cv2.imwrite(os.path.join(sqResultDir, 'epipoLine_sift_batch{}_img{}.png'.format(img_idx, idx)), vis)
        print("Writing image ... " + 'epipoLine_sift_batch{}_img{}.png'.format(img_idx, idx))
        images.append(vis)
        errors['batch{}_img{}_left'.format(img_idx, idx)] = l_avgErr
        errors['batch{}_img{}_right'.format(img_idx, idx)] = r_avgErr
    return np.array(images), errors



def training(args, model, device, trainLoader, optimizer, criterion, epoch, writer, allParamsRegularized= False):
    '''
    Training the model.
    :param args: input arguments
    :param model: training model
    :param device: device
    :param trainLoader: training loader
    :param optimizer: optimizer
    :param criterion: criterion
    :param epoch: current epoch
    :param writer: tensorboard writer
    :param allParamsRegularized: regularize all params?
    :return:
    '''
    # enter train mode
    model.train()
    # saving losses
    totalLoss = []
    print(50 * "*")
    print("Training Epoch ... ", epoch)
    print(50 * "*")

    # number of batches in the training dataset
    l = len(trainLoader)
    # length of the whole training dataset.
    L = len(trainLoader.dataset)
    # Regularizer
    reg_loss = L2Regularizer(model=model, lambda_reg=0.01)
    for batch_idx, (data, target, (_,_)) in enumerate(trainLoader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if allParamsRegularized:
            loss = reg_loss.regularized_all_param(reg_loss_function=loss)
        totalLoss.append(loss.item())

        loss.backward()

        optimizer.step()
        # writing the loss of the current batch to tensorboard
        writer.add_scalar('Batch Loss',
                          loss.data.cpu(),
                          epoch * l + batch_idx)
        if batch_idx % args.log_interval == args.log_interval-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), L ,
                       100. * (batch_idx+1) / l, loss.data.cpu()))

    # writing the loss of the current epoch to tensorboard
    writer.add_scalars('Train Epoch Loss', {'Training': np.mean(np.array(totalLoss))}, epoch)
    writer.flush()


def validating(args, model, device, valLoader, epoch, writer, log, trainloader=None):
    '''

    :param args:
    :param model:
    :param device:
    :param valLoader:
    :param epoch:
    :param writer:
    :param log:
    :param trainloader:
    :return:
    '''
    visualDir = "visualization/model_{}".format(args.exp)
    model.eval()
    totalLoss = []
    trainTotalLoss = []
    criterion = nn.MSELoss()

    print(50 * "*")
    print("Validating Epoch ... ", epoch)
    print(50 * "*")
    L = len(valLoader.dataset)
    with torch.no_grad():
        for id, (data, target, (left_img, right_img)) in enumerate(valLoader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)
            loss = criterion(output, target)
            totalLoss.append(loss.item())

            if id == 0:
                print("Visualize ... Batch {}".format(id))
                imageBatch, errors = visualize2Images(args, left_img, right_img, output, loss, epoch, id, visualizeDir=visualDir)
                log["epoch_{}_batch_{}".format(epoch, id)] = errors

                nameErr = 'batch{}_img{}_right'.format(id, 0)
                print(nameErr)
                writer.add_scalars("Testing Image Errors", {"exp_{}_{}".format(args.exp, nameErr): errors[nameErr]}, epoch)
                writer.add_images("VisualResult_Exp_{}".format(args.exp),  imageBatch, global_step=epoch, dataformats='NHWC')

        if trainloader:
            for id, (data, target, (_, _)) in enumerate(trainloader):
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                output = model(data)
                loss = criterion(output, target)
                trainTotalLoss.append(loss.item())

    valMean = np.mean(np.array(totalLoss))
    writer.add_scalars('Epoch Loss',{'Validate': valMean} ,epoch)
    if trainloader:
        trainMean = np.mean(np.array(trainTotalLoss))
        writer.add_scalars('Epoch Loss', {'Train': trainMean}, epoch)
        print('Trainning Set: Average Error: {:.6f}'.format(trainMean))
    writer.flush()
    print('Validation set: Average Error: {:.6f}. Length of set : {}.'.format(valMean, L))


def main():
    '''
    Running the models here
    :return:
    '''
    parser = argparse.ArgumentParser(description='DeepF_noCorrs')

    parser.add_argument('--deviceID', type=int, default=0, metavar='N',
                        help='The GPU ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='batch size for training set (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='batch size for testing set (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0000001, metavar='LR',
                        help='learning rate (default: 0.0000001)')
    parser.add_argument('--exp', type=int, default=0, metavar='experiment ID',
                        help='naming the experiment ID (default: 0)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many iterations to wait before printing the loss status')

    parser.add_argument('--k-fold', type=int, default=3, metavar='1-9',
                        help='how many folds in the test set (default: 3)')

    parser.add_argument('--kth', type=int, default=0, metavar='0-10',
                        help='The kth fold (default: 0)')

    parser.add_argument("--model", type=str, default='deepfmat', metavar="deepfmat, resnet, vgg16, alex",
                        help='Selecting the training models (default: deepfmat)')

    parser.add_argument("--norm", type=str, default='ETR', metavar='ETR, ABS, FBN',
                        help="Selecting the normalization method (default: ETR)")

    args = parser.parse_args()

    # -------------Dataset Path-----------------------------------
    POSES_PATH = "/media/slark/Data/Projects/dataset/data_kitti/dataset/poses"
    SEQUENCE_PATH = "/media/slark/Data/Projects/dataset/data_kitti/dataset/sequences"
    log_dir = os.path.join("logs/batchLosses/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    resultDir = os.path.join(ROOT_DIR, "analysis/experiment_{}_lr{}_batch{}".format(args.exp, args.lr, args.batch_size))
    # ------------------------------------------------------------

    # -------------Loading dataset--------------------------------
    db = DataSet(SEQUENCE_PATH, POSES_PATH)
    train, val, test = db.dataSets(k_fold=args.k_fold, kth=args.kth)
    poses = db.poses
    train_loader = DataLoader(dataSet=train
                              , poses=poses
                              , dType="train"
                              , camera=0)
    val_loader = DataLoader(dataSet=val
                            , poses=poses
                            , dType="validate"
                            , camera=0)
    # ------------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    device = torch.device(args.deviceID if use_cuda else "cpu")
    trainLoader = torch.utils.data.DataLoader(train_loader
                                              , batch_size=args.batch_size
                                              , shuffle=True
                                              , **kwargs)
    valLoader = torch.utils.data.DataLoader(val_loader
                                            , batch_size=args.test_batch_size
                                            , shuffle=False
                                            , **kwargs)

    resultModelFile = os.path.join(resultDir, log_dir, "result_{}".format(args.exp))

    print(50*"*")
    print("Running experiment {}".format(args.exp))
    print(50 * "*")

    outputSize = 9
    writer = SummaryWriter(os.path.join(resultDir, log_dir))
    if args.model == "deepfmat":
        model = DeepFMatNet(outputSize=outputSize, norm=args.norm).to(device)
    elif args.model == "alex":
        model = DeepFMatAlex(outputSize=outputSize, norm=args.norm).to(device)
    elif args.model == "vgg16":
        model = DeepFMatVGG16(outputSize=outputSize, norm=args.norm).to(device)
    elif args.model == "resnet":
        model = DeepFMatResNet18(outputSize=outputSize, norm=args.norm).to(device)

    if os.path.isfile(resultModelFile):
        try:
            model.load_state_dict(torch.load(resultModelFile))
        except:
            print("Cannot load the saved model")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    testErrorLog = os.path.join(resultDir, "log.txt")
    log = {}
    if os.path.isfile(testErrorLog):
        with open(testErrorLog, "rb") as f:
            log = pickle.load(f)
    for epoch in range(1, args.epochs+1):
        startTime = time.time()
        training(args, model, device, trainLoader, optimizer, criterion, epoch, writer)

        validating(args, model, device, valLoader, epoch, writer, log, trainloader=trainLoader)
        scheduler.step()
        torch.save(model.state_dict(), resultModelFile)
        torch.save(model.state_dict(), resultModelFile + "_epoch_{}".format(epoch))
        with open(testErrorLog, 'wb') as f:
            pickle.dump(log, f)
        endTime = time.time()
        writer.add_scalar('Time Epoch', endTime, epoch)
        print('--------{}--------\n'.format(endTime - startTime))
    writer.close()

if __name__ == '__main__':
    main()