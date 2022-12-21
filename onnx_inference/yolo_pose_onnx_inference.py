import os
import numpy as np
import cv2
import argparse
import onnxruntime
import torch
from torchvision import transforms
from tqdm import tqdm
from utils.general import xywh2xyxy


def letterbox(im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # 调整大小和垫图像，同时满足跨步多约束
    shape = im.shape[:2]  # current shape [height, width]
    new_shape = [640, 640]

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
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255
    return im, r, (dw, dh)


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path)

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]

    inp = {inname[0]: input}

    output = session.run(outname, inp)
    return output


def model_inference_image(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    im, ratio, dwdh = letterbox(image, auto=False)

    output = model_inference(model_path, im)
    post_process(img_path, output)


def post_process(img_file, output, nc=4, score_threshold=0.1):
    dims = output[0]
    det_bboxes = output[0][-1, :, 0:4]
    det_scores = output[0][-1, :, 4]
    det_labels = np.argmax(output[0][-1, :, 5:5 + nc], axis=1)
    kpts = output[0][-1, :, 5 + nc:]

    print(det_scores.max())
    img = cv2.imread(img_file)
    img = cv2.resize(img, (640, 640))

    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx].reshape(1, 4).astype(np.int)
        kpt = kpts[idx]
        x = xywh2xyxy(det_bbox)

        if det_scores[idx] > score_threshold:
            cv2.rectangle(img, (x[0][0], x[0][1]), (x[0][2], x[0][3]), (0, 0, 255),2)
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="../runs/train/exp2/weights/best.onnx")
    parser.add_argument("--img_path", type=str,
                        default="../data/image/2.jpg")
    opt = parser.parse_args()

    model_inference_image(opt.model_path, opt.img_path)
