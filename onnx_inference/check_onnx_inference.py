import argparse

import onnxruntime
import torch
from models.experimental import attempt_load

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def torch_ouput(weights, device, input):
    model = attempt_load(weights, map_location=device)
    pred = model(input)[0]
    output = pred.cpu().detach().numpy()
    return output


def onnx_output(model_path, input):
    session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    temp1 = session.get_outputs()[0].name
    temp2 = session.get_inputs()[0].name

    output = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: input})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str,
                        default="../runs/train/exp3/weights/best.pt",help='existing project/name ok, do not increment')
    parser.add_argument("--onnx_path", type=str,
                        default="../runs/train/exp3/weights/best.onnx",help='existing project/name ok, do not increment')

    opt = parser.parse_args()

    dummy_input = torch.randn(1, 3, 640, 640, device='cpu')
    temp1=torch_ouput(opt.pt_path, 'cpu', dummy_input)
    temp2=onnx_output(opt.onnx_path,dummy_input.cpu().detach().numpy())
    #在这里🦣断点开debug看
    print(temp1[:10])
    print(temp2[:10])



