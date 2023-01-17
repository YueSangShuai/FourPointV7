import onnx

onnx_model = onnx.load("../runs/train/exp3/weights/best.onnx")
outputs = onnx_model.graph.output
remove_id = []
# print(onnx_model.graph.output)

del outputs[1]
del outputs[1]
del outputs[1]

onnx.save(onnx_model,"deloutput.onnx")
print(onnx_model.graph.output)
