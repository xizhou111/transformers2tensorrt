onnx_path=./results/roberta_pretrain_512_static.onnx
output_engine=./results/roberta_pretrain_512.trt


python onnx2trt.py \
    --onnx ${onnx_path} \
    --output ${output_engine} \