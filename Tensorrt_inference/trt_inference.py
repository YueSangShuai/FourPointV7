import argparse
import copy
from operator import itemgetter

import cv2
import numpy as np
import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit

from onnx_inference.onnx_inference import IOU
from utils.datasets import letterbox as letterboxone
from utils.general import check_img_size, scale_coords
from onnx_inference.onnx_inference import letterbox as letterboxtwo


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


class TrtModel():
    '''
    TensorRT infer
    '''

    def __init__(self, trt_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            if binding == "output":
                self.idx = engine.get_binding_index(binding)
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def __call__(self, img_np_nchw):
        '''
        TensorRT推理
        :param img_np_nchw: 输入图像
        '''
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs[0], img_np_nchw.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[self.idx - 1], cuda_outputs[self.idx - 1], stream)
        stream.synchronize()
        self.ctx.pop()
        return host_outputs[self.idx - 1]

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()


def img_process(img_path, long_side=640, stride_max=32):
    '''
    图像预处理
    '''
    orgimg = cv2.imread(img_path)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterboxone(img0, new_shape=imgsz, auto=False)[0]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def video_process(img, long_side=640, stride_max=32):
    '''
    图像预处理
    '''
    orgimg = copy.deepcopy(img)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterboxone(img0, new_shape=imgsz, auto=False)[0]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def img_imshow(img_path, nc=1, kpts=17, conf_thresh=0.5, iou_thresh=0.5, kpts_threshold=0.5):
    img, orgimg = img_process(img_path)
    pred = model(img.numpy())  # forward
    rescults = np.split(pred, opt.output_shape[1])
    sorted_rescult = sorted(rescults, key=itemgetter(4), reverse=True)
    remove = np.zeros(len(rescults), int)
    rescult_data = []
    visual_img = letterboxtwo(orgimg, new_shape=img.shape[2:], auto=False)[0]

    # NMS
    for idx in range(len(sorted_rescult)):
        box_buff = sorted_rescult[idx]
        box1 = sorted_rescult[idx][:4]

        if box_buff[4] < conf_thresh:
            continue
        if remove[idx] == 1:
            continue
        rescult_data.append(box_buff)
        for jdx in range(idx + 1, len(sorted_rescult)):
            box_buff2 = sorted_rescult[jdx]
            box2 = sorted_rescult[jdx][:4]

            if box_buff2[4] < conf_thresh:
                continue
            if remove[jdx] == 1:
                continue
            if IOU(box1, box2) > iou_thresh:
                remove[jdx] = 1

    for idx in range(len(rescult_data)):
        temp = rescult_data[idx]
        det_bbox = rescult_data[idx][:4].reshape(1, 4).astype(np.int)
        label = np.argmax(rescult_data[idx][5:5 + nc])
        x = xywh2xyxy(det_bbox)
        cv2.rectangle(visual_img, (x[0][0], x[0][1]), (x[0][2], x[0][3]), (0, 0, 255), 2)
        det_kpts = rescult_data[idx][5 + nc:].reshape(1, 3 * kpts).astype(np.float)

        det_points = det_kpts[0][:2 * kpts]
        det_points_conf = det_kpts[0][2 * kpts:]

        cv2.putText(visual_img, str(label), (int(det_points[0]), int(det_points[4])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
        for jdx in range(len(det_points_conf)):
            temp = det_points_conf[jdx]

            if det_points_conf[jdx] > kpts_threshold:
                temp1 = int(det_points[jdx])
                temp2 = int(det_points[jdx + 4])
                cv2.circle(visual_img, (temp1, temp2), 5, (255, 0, 255), -1)

    cv2.imshow("rescult", visual_img)
    cv2.waitKey(3000)


def video_imshow(video_path, nc=1, kpts=17, conf_thresh=0.5, iou_thresh=0.5, kpts_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        img, orgimg = video_process(frame)
        pred = model(img.numpy())  # forward
        rescults = np.split(pred, opt.output_shape[1])
        sorted_rescult = sorted(rescults, key=itemgetter(4), reverse=True)
        remove = np.zeros(len(rescults), int)
        rescult_data = []
        visual_img = letterboxtwo(orgimg, new_shape=img.shape[2:], auto=False)[0]

        # NMS
        for idx in range(len(sorted_rescult)):
            box_buff = sorted_rescult[idx]
            box1 = sorted_rescult[idx][:4]

            if box_buff[4] < conf_thresh:
                continue
            if remove[idx] == 1:
                continue
            rescult_data.append(box_buff)
            for jdx in range(idx + 1, len(sorted_rescult)):
                box_buff2 = sorted_rescult[jdx]
                box2 = sorted_rescult[jdx][:4]

                if box_buff2[4] < conf_thresh:
                    continue
                if remove[jdx] == 1:
                    continue
                if IOU(box1, box2) > iou_thresh:
                    remove[jdx] = 1

        for idx in range(len(rescult_data)):
            temp = rescult_data[idx]
            det_bbox = rescult_data[idx][:4].reshape(1, 4).astype(np.int)
            label = np.argmax(rescult_data[idx][5:5 + nc])
            x = xywh2xyxy(det_bbox)
            cv2.rectangle(visual_img, (x[0][0], x[0][1]), (x[0][2], x[0][3]), (0, 0, 255), 2)
            det_kpts = rescult_data[idx][5 + nc:].reshape(1, 3 * kpts).astype(np.float)

            det_points = det_kpts[0][:2 * kpts]
            det_points_conf = det_kpts[0][2 * kpts:]

            cv2.putText(visual_img, str(label), (int(det_points[0]), int(det_points[4])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
            for jdx in range(len(det_points_conf)):
                temp = det_points_conf[jdx]

                if det_points_conf[jdx] > kpts_threshold:
                    temp1 = int(det_points[jdx])
                    temp2 = int(det_points[jdx + 4])
                    cv2.circle(visual_img, (temp1, temp2), 5, (255, 0, 255), -1)

        cv2.imshow("rescult", visual_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=r"../data/image/3.jpg", help='img path')
    parser.add_argument('--trt_path', type=str, default=r"../runs/train/exp3/weights/best.trt", help='trt_path')
    parser.add_argument('--output_shape', type=list, default=[1, 25200, 21],
                        help='input[1,3,640,640] ->  output[1,25200,16]')
    parser.add_argument('--video_path', default=r"../data/image/1.mp4",
                        help='using video')
    parser.add_argument("--nc", type=int, default=4, help="总类别")
    parser.add_argument("--kpts", type=int, default=4, help="总点数")
    parser.add_argument("--conf_thresholf", type=float, default=0.5, help="置信度")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="nms的iou阈值")
    parser.add_argument("--kpts_threshold", type=float, default=0.5, help="点的置信度")
    opt = parser.parse_args()
    model = TrtModel(opt.trt_path)

    if opt.img_path != "":
        img_imshow(img_path=opt.img_path, nc=opt.nc, kpts=opt.kpts, conf_thresh=opt.conf_thresholf,
                   iou_thresh=opt.iou_threshold, kpts_threshold=opt.kpts_threshold)
    if opt.video_path != "":
        video_imshow(video_path=opt.video_path, nc=opt.nc, kpts=opt.kpts, conf_thresh=opt.conf_thresholf,
                   iou_thresh=opt.iou_threshold, kpts_threshold=opt.kpts_threshold)
    model.destroy()