import argparse
import time

import onnxruntime
import torch
from tqdm import tqdm
import tensorrt as trt
import pycuda.driver as cuda
from models.experimental import attempt_load
import numpy as np


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
            # if binding == "output" or binding == "input":
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
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        return host_outputs[0]

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()


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
    return ((end - start) * 1000) / float(iter)


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
    total_time = run(model.to(opt.device), img.to(opt.device), opt.warmup_iter, opt.iter)
    rescult1 = model.to(opt.device)(img.to(opt.device))
    print('PT is  %.2f ms/img' % total_time)

    # -----------------------tensorrt-----------------------------------------
    model = TrtModel(opt.trt_path)
    total_time = run(model, img.numpy(), opt.warmup_iter, opt.iter)
    rescult2 = model(img)
    model.destroy()
    print('TensorRT is  %.2f ms/img' % total_time)
