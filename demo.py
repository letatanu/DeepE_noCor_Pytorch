from models.models import DeepFMatNet
import cv2
import torch
import argparse
import numpy as np
import os


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

def visualize(left_path, right_path, f_mat, sqResultDir):
    colors = [
        (255, 102, 102),
        (102, 255, 255),
        (125, 125, 125),
        (204, 229, 255),
        (0, 0, 204)
    ]
    THRESHOLD = 0.2
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    f_mat = np.array(f_mat.reshape((3, 3)))

    left_img = cv2.imread(left_path)
    # -------
    hl, wl = left_img.shape[0], left_img.shape[1]
    left_img = left_img[int(hl / 2) - 128: int(hl / 2) + 128, int(wl / 2) - 128: int(wl / 2) + 128]
    # --------------------------

    left_imgG = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
    left_img_line = left_img.copy()

    right_img = cv2.imread(right_path)
    # -------------------------------
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

    img3 = cv2.drawMatchesKnn(right_imgG, kps_left, right_imgG, kps_right, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(sqResultDir, 'feature_matching.png'), img3)

    err_l = []
    err_r = []
    img_W = left_img.shape[1] - 1
    # ---------------------------------------------------------------------
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
    x, y, w, h = 0, 0, 50, 25

    # Draw black background rectangle
    cv2.rectangle(vis, (7, 10), (w, h), (0, 0, 0), -1)
    cv2.rectangle(vis, (7, img_H-20), (w, img_H-7), (0, 0, 0), -1)


    cv2.putText(vis, '{:.4f}'.format(float(l_avgErr)), (10, 20), font, 0.3, color=(255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(vis, '{:.4f}'.format(float(r_avgErr)), (10, img_H - 10), font, 0.3, color=(255, 255, 255),
                lineType=cv2.LINE_AA)


    if not os.path.exists(sqResultDir):
        os.makedirs(sqResultDir)
    print("Writing image ... " + 'epipoLine_sift.png')
    cv2.imwrite(os.path.join(sqResultDir, 'epipoLine_sift.png'), vis)

def inputProcessing(left_Img, right_Img):
    size = 64

    hl, wl = left_Img.shape[0], left_Img.shape[1]
    left_Img = left_Img[int(hl / 2) - int(size / 2): int(hl / 2) + int(size / 2),
               int(wl / 2) - int(size / 2): int(wl / 2) + int(size / 2)]

    hr, wr = right_Img.shape[0], right_Img.shape[1]

    right_Img = right_Img[int(hr / 2) - int(size / 2): int(hr / 2) + int(size / 2),
                int(wr / 2) - int(size / 2): int(wr / 2) + int(size / 2)]

    left_Img = (left_Img - 127.5) / 127.5
    right_Img = (right_Img - 127.5) / 127.5

    left_Img = np.expand_dims(left_Img, axis=2)
    right_Img = np.expand_dims(right_Img, axis=2)
    input = np.concatenate((left_Img, right_Img), axis=2)
    input = np.rollaxis(input, 2, 0)
    return input


def main():
    parser = argparse.ArgumentParser(description='DeepF_noCorrs')
    parser.add_argument('--deviceID', type=int, default=0, metavar='N',
                        help='The GPU ID (default: 0)')

    parser.add_argument("--norm", type=str, default='ETR', metavar='ETR, ABS, FBN',
                        help="Select the normalization method (default: ETR)")
    args = parser.parse_args()

    img1P = "examples/000005.png"
    img2P = "examples/000000.png"
    outputSize = 9
    resultModelFile = "result_0"
    img1 = cv2.imread(img1P, flags=cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2P, flags=cv2.IMREAD_GRAYSCALE)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.deviceID if use_cuda else "cpu")
    model = DeepFMatNet(outputSize=outputSize, norm=args.norm).to(device)

    if os.path.isfile(resultModelFile):
        try:
            model.load_state_dict(torch.load(resultModelFile))
        except:
            print("Cannot load the saved model")
    model.eval()
    with torch.no_grad():
        input = inputProcessing(img1, img2)
        input = torch.from_numpy(input).to(device, dtype=torch.float)
        input =  torch.unsqueeze(input, 0)
        f_mat = model(input)
        f_mat = f_mat.cpu().numpy()
        visualize(img1P, img2P, f_mat, sqResultDir="visualization")

if __name__ == '__main__':
    main()
