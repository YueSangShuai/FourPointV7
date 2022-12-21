import os
import numpy as np
import cv2
import argparse
import onnxruntime
import torch
from torchvision import transforms
from tqdm import tqdm


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def pre_img(img_file, img_mean=127.5, img_scale=1 / 127.5):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    image = img.astype(np.float32) / 255.0
    image = image.transpose((2, 0, 1))  # (3, 96, 96)
    image = image[np.newaxis, :, :, :]  # (1, 3, 96, 96)
    image = np.array(image, dtype=np.float32)

    return image


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path)

    output = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: input})
    return output


def model_inference_image(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    img = pre_img(img_path)
    output = model_inference(model_path, img)
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
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]

        if det_scores[idx] > score_threshold:
            cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])), (0, 255, 0),
                          2)
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/yuesang/Project/PycharmProjects/FourPointV7/runs/train/exp/weights/best.onnx")
    parser.add_argument("--img_path", type=str,
                        default="/home/yuesang/Project/PycharmProjects/yolov7-FourPoint/data/image/1.jpg")
    opt = parser.parse_args()

    model_inference_image(opt.model_path, opt.img_path)
