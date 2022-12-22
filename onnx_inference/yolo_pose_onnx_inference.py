import os
import time
from operator import itemgetter

import numpy as np
import cv2
import argparse
import onnxruntime
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from utils.general import xywh2xyxy


def IOU(box1, box2):
    """
    :box1:[x1,y1,x2,y2]# (x1,y1)表示左上角，(x2,y2)表示右下角
    :box2:[x1,y1,x2,y2]
    :return: iou_ratio交并比
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    xmax = max(box1[0], box1[2], box2[0], box2[2])
    ymax = max(box1[1], box1[3], box2[1], box2[3])
    xmin = min(box1[0], box1[2], box2[0], box2[2])
    ymin = min(box1[1], box1[3], box2[1], box2[3])
    W = xmin + width1 + width2 - xmax
    H = ymin + height1 + height2 - ymax
    if W <= 0 or H <= 0:  # 当H和W都小于等于0的时候没有交集，其他情况有交集。
        iou_ratio = 0
    else:  # 其他情况有交集
        iou_area = W * H  # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area)  # 并集的面积
    return iou_ratio


def letterbox(im, color=(114, 114, 114), new_shape=[640, 640], auto=True, scaleup=True, stride=32):
    # 调整大小和垫图像，同时满足跨步多约束
    shape = im.shape[:2]  # current shape [height, width]

    # 如果是单个size的话，就在这里变成一双
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 尺度比 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不扩大(为了更好的val mAP)
        r = min(r, 1.0)

    # 计算填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # 最小矩形区域
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    image = im.copy()
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255
    return image, im, r, (dw, dh)


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    # session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]

    inp = {inname[0]: input}

    output = session.run(outname, inp)
    return output


def post_process(imshow_img, output,kpts=17,conf_threshold=0.3, iou_threshold=0.5, kpts_threshold=0.5,videomode=False):
    remove = np.zeros(output[0].shape[1], int)
    rescult_data = []
    sout_output = sorted(output[0][0], key=itemgetter(4), reverse=True)

    # TODO:nms
    for idx in range(len(sout_output)):
        box_buff = sout_output[idx]
        box1 = sout_output[idx][:4]

        if box_buff[4] < conf_threshold:
            continue
        if remove[idx] == 1:
            continue
        rescult_data.append(box_buff)
        for jdx in range(idx + 1, len(sout_output)):
            box_buff2 = sout_output[jdx]
            box2 = sout_output[jdx][:4]

            if box_buff2[4] < conf_threshold:
                continue
            if remove[jdx] == 1:
                continue
            if IOU(box1, box2) > iou_threshold:
                remove[jdx] = 1

    # TODO:show_rescult
    for idx in range(len(rescult_data)):
        det_bbox = rescult_data[idx][:4].reshape(1, 4).astype(np.int)
        x = xywh2xyxy(det_bbox)
        cv2.rectangle(imshow_img, (x[0][0], x[0][1]), (x[0][2], x[0][3]), (0, 0, 255), 2)
        det_kpts = rescult_data[idx][-3*kpts:].reshape(1, 3*kpts).astype(np.float)

        det_points=det_kpts[0][:2*kpts]
        det_points_conf=det_kpts[0][2*kpts:]

        for jdx in range(len(det_points_conf)):
            temp=det_points_conf[jdx]
            if det_points_conf[jdx]>kpts_threshold:
                temp1=int(det_points[jdx])
                temp2=int(det_points[jdx+4])
                cv2.circle(imshow_img,(temp1,temp2),5,(255,0,255),-1)


        temp=det_kpts

    if not videomode:
        cv2.imshow('result', imshow_img)
        cv2.waitKey(1000)
    else:
        cv2.imshow('result', imshow_img)
        cv2.waitKey(1)


def model_inference_image(model_path, img_path=None,kpts=17,image_size=[640, 640], conf_threshold=0.3, iou_threshold=0.5,kpts_threshold=0.5):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    imshow_img, im = letterbox(image, new_shape=image_size, auto=False)[:2]

    output = model_inference(model_path, im)
    post_process(imshow_img, output, kpts=kpts,conf_threshold=conf_threshold, iou_threshold=iou_threshold,kpts_threshold=kpts_threshold)


def model_inference_video(model_path=None, video_path=None, kpts=17,image_size=[640, 640], conf_threshold=0.3,
                          iou_threshold=0.5,kpts_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = img.copy()
        imshow_img, im = letterbox(image, new_shape=image_size, auto=False)[:2]
        output = model_inference(model_path, im)
        post_process(imshow_img, output, kpts=kpts,conf_threshold=conf_threshold, iou_threshold=iou_threshold, videomode=True,kpts_threshold=kpts_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../runs/train/exp3/weights/best.onnx",help="onnx模型位置")
    parser.add_argument("--img_size", type=tuple, default=[640, 640],help="图像大小")
    parser.add_argument("--img_path", type=str, default="../data/image/4.jpg",help="图片路径")
    parser.add_argument("--video_path", type=str, default="../data/image/1.mp4",help="视频路径")
    parser.add_argument("--nc", type=int, default=4,help="总类别")
    parser.add_argument("--kpts", type=int, default=4,help="总点数")
    parser.add_argument("--conf_thresholf", type=float, default=0.5,help="置信度")
    parser.add_argument("--iou_threshold", type=float, default=0.5,help="nms的iou阈值")
    parser.add_argument("--kpts_threshold", type=float, default=0.5,help="点的置信度")
    opt = parser.parse_args()

    if not opt.img_path == "":
        model_inference_image(model_path=opt.model_path, img_path=opt.img_path,kpts=opt.kpts,image_size=opt.img_size, conf_threshold=opt.conf_thresholf, iou_threshold=opt.iou_threshold,kpts_threshold=opt.kpts_threshold)
    if not opt.video_path == "":
        model_inference_video(model_path=opt.model_path, video_path=opt.video_path,kpts=opt.kpts, image_size=opt.img_size, conf_threshold=opt.conf_thresholf, iou_threshold=opt.iou_threshold,kpts_threshold=opt.kpts_threshold)
