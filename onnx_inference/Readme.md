# ***ONNX推理***

该目录包括onnx的推理，onnx和pt的结果对比以及给onnx模型添加预处理部分   
带预处理的推理部分暂时没有写，推理部分只有不带预处理的
结果对比需要开debug和断电看最终两个模型推理出来的结果   
onnx_dadpre：onnx添加预处理操作为后续Tensorrt推理做基础  
onnx_remove：可能是我训练没有改好导致最后转出来的onnx文件有四个输出删除三个用不到的输出  
onnx_inference:用来推理pt转onnx后得到的原始模型
