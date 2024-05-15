onnx_path=./results/roberta_static.onnx
output_engine=./results/roberta_py2trt_test.trt


python onnx2trt.py \
    --onnx ${onnx_path} \
    --output ${output_engine} \