#include <Python.h>


#include <fstream>

#include <iostream>
#include <sstream>

#include "numpy/arrayobject.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>



#include "opencv2/ximgproc.hpp"

//#include "module.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"
#include "D:\opencv-4.2.0\opencv-4.2.0\modules\python\src2\\pycompat.hpp"


#include <time.h>
#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>


# include <opencv2/dirent.h>

#include "opencv2/ximgproc.hpp"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

using namespace std;
//
//void
//main(int argc, char** argv)
//{
//    // initialize Python
//    Py_Initialize();
//
//    // compile our function
//    stringstream buf;
//    buf << "def add( n1 , n2 ) :" << endl
//        << "    return n1+n2" << endl;
//    PyObject* pCompiledFn = Py_CompileString(buf.str().c_str(), "", Py_file_input);
//    assert(pCompiledFn != NULL);
//
//    // create a module        
//    PyObject* pModule = PyImport_ExecCodeModule("test", pCompiledFn);
//    assert(pModule != NULL);
//
//    // locate the "add" function (it's an attribute of the module)
//    PyObject* pAddFn = PyObject_GetAttrString(pModule, "add");
//    assert(pAddFn != NULL);
//
//
//    // create a new tuple with 2 elements
//    PyObject* pPosArgs = PyTuple_New(2);
//
//    // convert the first command-line argument to an int, then put it into the tuple
//    PyObject* pVal1 = PyLong_FromString("1", NULL, 10);
//    assert(pVal1 != NULL);
//    int rc = PyTuple_SetItem(pPosArgs, 0, pVal1); // nb: tuple position 0
//    assert(rc == 0);
//
//    // convert the second command-line argument to an int, then put it into the tuple
//    PyObject* pVal2 = PyLong_FromString("1000", NULL, 10);
//    assert(pVal2 != NULL);
//    rc = PyTuple_SetItem(pPosArgs, 1, pVal2); // nb: tuple position 1
//    assert(rc == 0);
//
//
//    // create a new dictionary 
//    PyObject* pKywdArgs = PyDict_New();
//    assert(pKywdArgs != NULL);
//
//    // call our function 
//    PyObject* pResult = PyObject_Call(pAddFn, pPosArgs, pKywdArgs);
//    assert(pResult != NULL);
//
//
//    // convert the result to a string 
//    PyObject* pResultRepr = PyObject_Repr(pResult);
//    cout << "The answer: " << PyUnicode_AsUTF8(pResultRepr) << endl;
//
//
//
//
//
//    // clean up
//    Py_DecRef(pAddFn);
//    Py_DecRef(pModule);
//    Py_DecRef(pCompiledFn);
//    Py_Finalize();
//}









int load_Data(string dir, vector<string>& files)
{
    DIR* dp;
    struct dirent* dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (string(dirp->d_name).find(".jpg") != string::npos)
            files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    sort(files.begin(), files.end());
    return 0;
}







PyObject* pName;
int  main(int argc, char** argv)
{
    // initialize Python
    Py_Initialize();

    //PyRun_SimpleString("from time import time,ctime\n"
    //    "print('Today is',ctime(time()))\n");
    ////Run a simple file
    //FILE* PScriptFile = fopen("OCD/newbefore.py", "r");
    //if (PScriptFile) {
    //    std::cout << "run" << endl;
    //    PyRun_SimpleFile(PScriptFile, "OCD/newbefore.py");
    //    fclose(PScriptFile);
    //}





    cv::Mat img = cv::imread("C:\\Users\\ZahRa\\source\\repos\\EmbededPythonInC++ByObject\\EmbededPythonInC++ByObject\\1.jpg");

    // compile our function

string script1= R""""(

"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import time
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
print(torch.__path__)
print("hereeeeeee in new before ")
print(torch.__version__)
print(torch.__path__)
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
sys.path.append('OCD/')

import numpy as np
from skimage import io
import cv2
import json
import zipfile

#from craft import CRAFT

from collections import OrderedDict

import numpy as np
import cv2
import math

#craft

import torch
import torch.nn as nn
import torch.nn.functional as F

from basenet.vgg16_bn import vgg16_bn, init_weights

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

#model = CRAFT(pretrained=True).cuda()
#output, _ = model(torch.randn(1, 3, 768, 768).cuda())
#print(output.shape)






#refinenet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from basenet.vgg16_bn import init_weights


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = torch.cat([y.permute(0,3,1,2), upconv4], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)
#craft_util

# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap0, linkmap0, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap0.copy()
    textmap = textmap0.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    cv2.imshow("text_score", text_score)
    cv2.waitKey(1)

    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # text_score_comb = np.clip(text_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []

    areas = [stats[k, cv2.CC_STAT_AREA] for k in range(1, nLabels)]
    if len(areas) > 0:
        max_area_ind = np.argmax(np.array([stats[k, cv2.CC_STAT_AREA] for k in range(1, nLabels)])) + 1

        # print(nLabels, labels, stats, centroids)
        for k in range(1, nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if k != max_area_ind:
                continue
            if size < 10:
                continue

            # thresholding
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels==k] = 255
            segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0 : sx = 0
            if sy < 0 : sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:,0]), max(np_contours[:,0])
                t, b = min(np_contours[:,1]), max(np_contours[:,1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4-startidx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys




)"""";

        string script2 = R""""(



#imgproc
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


#print(len(sys.argv))
#print(sys.argv)

#parser = argparse.ArgumentParser(description='CRAFT Text Detection')
#parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
#parser.add_argument('--text_threshold', default=0.5, type=float, help='text confidence threshold')
#parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
#parser.add_argument('--link_threshold', default=0.3, type=float, help='link confidence threshold')
#parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
#parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
#parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
#parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
#parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')
#parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
#parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

 

#args = parser.parse_args()


""" For test images in a folder """
#image_list, _, _ = file_utils.get_files('D:/Plate_Detection/Plate photos/2/output')

result_folder = 'OCD/result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)





def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


      # load net
net = CRAFT()     # initialize
torch.set_grad_enabled(False)
if True:
     net.load_state_dict(copyStateDict(torch.load('OCD/weights/craft_mlt_25k.pth')))
else:
     net.load_state_dict(copyStateDict(torch.load('OCD/weights/craft_mlt_25k.pth', map_location='cpu')))

if True:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
	
net.eval()

    # LinkRefiner
refine_net = None
if True:
    #from refinenet import RefineNet
    refine_net = RefineNet()
    if True:
        refine_net.load_state_dict(copyStateDict(torch.load('OCD/weights/craft_refiner_CTW1500.pth')))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load('OCD/weights/craft_refiner_CTW1500.pth', map_location='cpu')))


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    # resize
    
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    x = x.contiguous()



    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # refine link
    start_time = time.time()
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    #print("----++++++++++++++++++++++++++++++++++++++++++-- %s seconds ---" % (time.time() - start_time))

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text




def saveResult(img, boxes, mask_file, dirname='./result/', verticals=None, texts=None):

        img = np.array(img)
        # img = np.zeros(img.shape, dtype=np.uint8)

        # make result file list
        #filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        # res_file = dirname + filename + '.txt'
        #res_img_file = dirname + filename + '.jpg'

        #if not os.path.isdir(dirname):
         #   os.mkdir(dirname)

        box_exits = False
        # with open(res_file, 'w') as f:
        if(len(boxes)):
            box_exits = True
            poly = np.array(boxes[0]).astype(np.int32).reshape((-1))

            poly = poly.reshape(-1, 2)
            print(poly)
            poly = find_corners(poly)
            print(poly)
            #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


        # Save result image
        # cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, useHarrisDetector=True)
        if box_exits:
            print(img.shape)
            #print(res_img_file)
            finalImage=perspective_trans(img, poly)
            #cv2.imwrite(res_img_file, finalImage)
            print(finalImage.shape)
            cv2.imshow("ggggggg",finalImage)
            cv2.waitKey(1)
            return bytearray(finalImage)
            return finalImage
            #blur = cv2.GaussianBlur(cv2.cvtColor(finalImage, cv2.COLOR_RGB2GRAY), (5, 5), 0)
            #ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #cv2.imwrite('D:/Plate_Detection/Plate photos/2/outth/0.jpg', th2)
            #cv2.imshow("plate_thresh", th2)
            #cv2.waitKey(1)
            #mask_addr = "C:/Users/ZahRa/source/repos/PlateDetection - 2012/PlateDetection/OCD/heat_maps/0" + '.jpg'
            # print(mask_file.shape)
            #cv2.imwrite(mask_addr, perspective_trans(mask_file, poly, resize=True, size=img.shape))



def perspective_trans(img, poly, max_width=350, max_height=100, resize=False, size=None):
    img0 = np.copy(img)
    cv2.imshow("heat map before perspective",img)
    cv2.waitKey(1)
    if resize:
        img0 = cv2.resize(img0, (size[1], size[0]))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    return cv2.warpPerspective(img0, M, (max_width, max_height))




def find_corners(poly):
    if len(poly) <= 4:
        return poly
    box = []
    box.append(poly[0])

    if(((float(poly[1][1]) - float(poly[0][1]))==0) and ((float(poly[1][0]) - float(poly[0][0]))==0)):
        prev_slope = 1
    elif(float(poly[1][1]) - float(poly[0][1]) == 0):
        prev_slope =(float(poly[1][1]) - float(poly[0][1])) * 1000
    elif(float(poly[1][0]) - float(poly[0][0])==0): 
        prev_slope =57;
    else:  
        prev_slope = (float(poly[1][1]) - float(poly[0][1])) / (float(poly[1][0]) - float(poly[0][0]))

    for i in range(1, len(poly)):


        if (((float(poly[i][1]) - float(poly[i-1][1]))==0) and ((float(poly[i][0]) - float(poly[i-1][0]))==0)):
            slope = 1
        elif(float(poly[i][1]) - float(poly[i-1][1]) == 0):
            slope =(float(poly[i-1][1]) - float(poly[i-1][1])) * 1000
        elif(float(poly[i][0]) - float(poly[i-1][0])==0): 
            slope =57;
        else:  
            slope = (float(poly[i][1]) - float(poly[i-1][1])) / (float(poly[i][0]) - float(poly[i-1][0]))

        if abs(prev_slope - slope) > 2:
            box.append(poly[i - 1])
        prev_slope = slope

    box.append(poly[-1])
    return np.array(box)




def load_data2(inputimage):
    


    print("here in test-image_list==================================================================")
    print("module:2 - img")
    image = inputimage
    print("here1")
        
    bboxes, polys, score_text = test_net(net, image, 0.5, 0.3, 0.3, True, True, refine_net)
    print("here2")
        
    # save score text
    #filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = "heat_maps/" + filename + '.jpg'
    # cv2.imwrite(mask_file, score_text)

    imm=saveResult(image, polys, score_text, dirname=result_folder)
    return imm
    print("here3")
   
)"""";

        string script = script1 + script2;
    PyObject* pCompiledFn = Py_CompileString(script.c_str(), "", Py_file_input);
    assert(pCompiledFn != NULL);



    cout << "hh";
    // create a module        
    PyObject* pModule = PyImport_ExecCodeModule("test", pCompiledFn);
    assert(pModule != NULL);

    if (pModule == NULL)
    {
        PyErr_Print();
        return 0;
    }



    // locate the "add" function (it's an attribute of the module)
    PyObject* pAddFn = PyObject_GetAttrString(pModule, "load_data2");
    assert(pAddFn != NULL);
    if (pAddFn == NULL)
    {
        PyErr_Print();
        return 0;
    }



    // create a new tuple with 1 elements
    PyObject* pPosArgs = PyTuple_New(1);


    string video_path = string("D:\\Plate_Detection\\Plate photos\\2");
    vector<string> files = vector<string>();
    load_Data(video_path, files);
    cv::Mat input_image;
    npy_intp dimensions[3];
    PyObject* pVal1;
    int rc;
    PyObject* pKywdArgs;
    PyObject* pResult;
    uchar* data;
    for (int frame = 1; frame < files.size(); frame++) {
        std::cout << "************************    " << frame << "   **********************************" << endl;
        clock_t tStart = clock();
        string imgName = files[frame];
        string frameName = video_path + "\\" + imgName;
        input_image = cv::imread(frameName, 1);
        img = input_image;
        // convert the first command-line argument to an int, then put it into the tuple
        import_array();
        dimensions[0] =  img.rows ;
        dimensions[1] = img.cols;
        dimensions[2] = img.channels();

        pVal1 = PyArray_SimpleNewFromData(img.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, img.data);


        //PyObject* pVal1 = PyLong_FromString("hi", NULL, 10);
        //assert(pVal1 != NULL);
         rc = PyTuple_SetItem(pPosArgs, 0, pVal1); // nb: tuple position 0
        assert(rc == 0);

        // create a new dictionary 
         pKywdArgs = PyDict_New();
        assert(pKywdArgs != NULL);
        if (pKywdArgs == NULL)
        {
            PyErr_Print();
            return 0;
        }
        // call our function 
         pResult = PyObject_Call(pAddFn, pPosArgs, pKywdArgs);
        assert(pResult != NULL);
        if (pResult == NULL)
        {
            PyErr_Print();
            return 0;
        }

        data = (uchar*)PyByteArray_AsString(pResult);
        cv::Mat imgg(100, 350, CV_8UC3, data);
        imshow("imgg", imgg);
        cv::waitKey(1);

        printf("Time-tStart_EACH_PLATE for each plate-------------- taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);














        //PyObject* product = PyObject_CallFunction(pAddFn, , numberPair.x, numberPair.y);
        //if (product != NULL)
        //{
        //    std::cout << "Product is " << PyLong_AsLong(product) << '\n';
        //    Py_DECREF(product);
        //}





        //cout << "image:" << frameName << endl;
        //imshow("input_imageeeeeeeeeeeeeeee", input_image);
        //cv::waitKey(0);
    }





    //PyArrayObject* contig = (PyArrayObject*)PyArray_FromAny(pResult,
    //    PyArray_DescrFromType(NPY_UINT8),
    //    0, 0, NPY_ARRAY_CARRAY, NULL);
    //assert(contig && PyArray_DIM(contig, 2) == 3);
    //if (contig == nullptr) {
    //    // Throw an exception
    //    PyErr_Print();
    //    return 0;
    //}

    //cv::Mat mat(PyArray_DIM(contig, 0), PyArray_DIM(contig, 1), CV_8UC3,
    //    PyArray_DATA(contig));

    //imshow("mat", mat);
    //cv::waitKey(0);
   


    //report string 
    //PyObject* pResultRepr = PyObject_Repr(pResult);
    //cout << "The answer: " << PyUnicode_AsUTF8(pResultRepr) << endl;





    // clean up
    Py_DecRef(pAddFn);
    Py_DecRef(pModule);
    Py_DecRef(pCompiledFn);
    Py_Finalize();

    return 0;
};