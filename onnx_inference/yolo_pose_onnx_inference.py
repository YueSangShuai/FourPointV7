import os
import numpy as np
import cv2
import argparse
import onnxruntime
import torch
from tqdm import tqdm


def pre_img(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    img = img/255.0
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)


    return img


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path)
    output = session.run([], {session.get_inputs()[0].name: input})
    return output


def model_inference_image(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    img = pre_img(img_path)
    output = model_inference(model_path, img)
    post_process(img_path, output, 4, 4)


def post_process(img_file, output, nc, kpt_point, score_threshold=0.5):
    dims=output[0]
    det_bboxes = output[0][-1, :, 0:4]
    det_scores = output[0][-1, :, 4]
    det_labels = np.argmax(output[0][-1, :, 5:5 + nc], axis=1)
    kpts = output[0][-1, :, 5 + nc:]

    img = cv2.imread(img_file)

    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]

        if det_scores[idx] > score_threshold:
            cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])), (0, 255, 0), 2)
    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/home/yuesang/Project/PycharmProjects/yolov7-FourPoint/runs/train/exp/weights/best.onnx")
    parser.add_argument("--img_path", type=str,
                        default="/home/yuesang/Project/PycharmProjects/yolov7-FourPoint/data/image/1.jpg")
    parser.add_argument("--display_rescult", action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    model_inference_image(opt.model_path, opt.img_path)
