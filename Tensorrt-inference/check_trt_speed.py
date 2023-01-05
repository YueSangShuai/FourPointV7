import argparse
import time

import onnxruntime
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from trt_model import TrtModel
from onnx_inference.check_onnx_inference import onnx_output


def run(model, img, warmup_iter, iter):
    print('start warm up...')
    for _ in tqdm(range(warmup_iter)):
        model(img)

    print('start calculate...')
    torch.cuda.synchronize()
    start = time.time()
    for __ in tqdm(range(iter)):
        model(img)
        torch.cuda.synchronize()
    end = time.time()
    return ((end - start) * 1000) / float(iter),model(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt_path", type=str, default="../runs/train/exp3/weights/best.trt",
                        help='existing project/name ok, do not increment')
    parser.add_argument("--torch_path", type=str, default="../runs/train/exp3/weights/best.pt",
                        help='existing project/name ok, do not increment')
    parser.add_argument("--img_shape", type=list, default=[1, 3, 640, 640],
                        help='existing project/name ok, do not increment')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--warmup_iter', type=int, default=100, help='warm up iter')
    parser.add_argument('--iter', type=int, default=300, help='average elapsed time of iterations')
    opt = parser.parse_args()

    # ----------------------torch-----------------------------------------
    img = torch.randn(opt.img_shape)
    model = attempt_load(opt.torch_path, map_location=torch.device('cuda'))  # load FP32 model
    model.eval()
    total_time,rescult1 = run(model.to(opt.device), img.to(opt.device), opt.warmup_iter, opt.iter)
    print('PT is  %.2f ms/img' % total_time)

    # -----------------------tensorrt-----------------------------------------
    model = TrtModel(opt.trt_path)
    total_time,rescult2 = run(model, img.numpy(), opt.warmup_iter, opt.iter)
    model.destroy()
    print('TensorRT is  %.2f ms/img' % total_time)
