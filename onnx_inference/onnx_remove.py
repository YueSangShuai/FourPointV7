import onnx

onnx_model = onnx.load("../runs/train/exp3/weights/best.onnx")
output = onnx_model.graph.output
remove_id = []
for node_id, node in enumerate(onnx_model.graph.node):
    if node.output[0] == "onnx::Split_478":
        remove_id.append(node_id)
    if node.output[0] == "onnx::Split_592":
        remove_id.append(node_id)
    if node.output[0] == "onnx::Split_705":
        remove_id.append(node_id)
for i in range(len(remove_id)):
    old_squeeze_node = onnx_model.graph.node[i]
    onnx_model.graph.node.remove(old_squeeze_node)

onnx.save(onnx_model,"deloutput.onnx")