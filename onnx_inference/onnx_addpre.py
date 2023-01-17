# coding=utf-8

"""
如果我们想将预处理加到onnx头部
1. 我们首先要导出预处理的onnx
2. 然后合并两个onnx

注意此时的输入就变成了 [n h w c]
"""
import argparse
import os

import torch
import onnx
import onnx.helper as helper

"""
yolov7-pose预处理只有 BGR->RGB   HWC->CHW  /255   

"""


class Preprocess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 输入的为uint8的NHWC形式

    def forward(self, x):
        # x原本是uint8的先转为float
        x = x[..., [2, 1, 0]]  # 产生了Gather节点。  BGR->RGB
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0

        return x


def getMeronnx(opt):
    pre = Preprocess()
    # 这里输入名字，尽量自定义，后面转trt可控
    torch.onnx.export(pre, (torch.zeros((1, opt.img_size, opt.img_size, 3), dtype=torch.float),), "./pre.onnx",
                      input_names=["input"])

    pre = onnx.load("./pre.onnx")
    model = onnx.load(opt.onnx_weights)

    # 先把pre模型名字加上前缀
    for n in pre.graph.node:
        if not n.name == "input":
            n.name = f"{n.name}"
            for i in range(len(n.input)):  # 一个节点可能有多个输入
                n.input[i] = f"{n.input[i]}"
            for i in range(len(n.output)):
                n.output[i] = f"{n.output[i]}"

    # 2 修改另一个模型的信息
    # 查看大模型的第一层名字
    for n in model.graph.node:
        if n.name == "/model.0/conv/Conv":
            n.input[0] = pre.graph.output[0].name

    for n in pre.graph.node:
        model.graph.node.append(n)

    # 还要将pre的输入信息 NHWC等拷贝到输入
    model.graph.input[0].CopyFrom(pre.graph.input[0])
    # 此时model的输入需要变为 pre的输入 pre/0
    model.graph.input[0].name = pre.graph.input[0].name

    save_weight_list = opt.onnx_weights.split('/')[:-1]
    save_path = ""
    for path in save_weight_list:
        save_path = os.path.join(save_path, path)
    save_path = os.path.join(save_path, "merge.onnx")

    onnx.save(model, save_path)
    os.unlink("./pre.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_weights', type=str,
                        default='../onnx_inference/deloutput.onnx',
                        help='initial weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()

    getMeronnx(opt)
