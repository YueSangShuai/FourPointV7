# ***Tensorrt推理***
trt模型可以用官方的trtexec进行onnx的转换  
trt模型推理的预处理部分我写的像坨狗屎所以导致整体帧率不高，懒得改了后面会写c++版本的推理用python看一下模型推理的结果而已  
trt_inference：采用的是官方的onnx转Tensorrt，其中输出有四层可以使用check_trt_model.py这个脚本查看  
trt_inferenceV2：将其中的三个无用的检测层删除，具体模型的转换之后会放在c++版本的推理中（目前已在python中实现onnx删除三个无用层详细见onnx_inference中的Readme） 
trt_inferenceV3:在删除三个无用检测层的基础之上，给trt模型添加预处理操作后的模型推理

