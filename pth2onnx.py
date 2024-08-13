# -*- coding: utf-8 -*-
# pth2onnx.py
import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

def export_static_onnx_model(text, model, tokenizer, static_onnx_model_path):
    # example_tensor
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True,max_length=256, padding="max_length", return_attention_mask=True)
    device = torch.device("cpu")
    inputs = inputs.to(device)
    print(inputs)

    with torch.no_grad():
        torch.onnx.export(model,            # model being run
                    (inputs['input_ids'],   # model input (or a tuple for multiple inputs)
                    inputs['attention_mask']),
                    static_onnx_model_path, # where to save the model (can be a file or file-like object)
                    verbose=True,
                    opset_version=19,       # the ONNX version to export the model to
                    input_names=['input_ids',    # the model's input names
                                'input_mask'],
                    output_names=['output'])     # the model's output names
        print("ONNX Model exported to {0}".format(static_onnx_model_path))

if __name__ == '__main__':
    torch_model = torch.load("./results/roberta_pretrain_512.pth")  # pytorch模型加载
    # 模型多次调用了tokenizer和model，待优化
    checkpoint = "/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_pretrain_512/checkpoint-30000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # transformers模型加载
    model.load_state_dict(torch_model)

    text = "如图,姐弟两人吃一个蛋糕。比较发现,(     )。\n姐姐：“我先吃这个蛋糕的 $$ \\frac{1}{3} $$。”\n弟弟：“我吃剩下的一半。”\nA. 姐姐吃得多  \nB. 弟弟吃得多  \nC. 两人吃得一样多\nD. 无法比较\n"
    static_onnx_model_path = "./results/roberta_pretrain_512_static.onnx"
    export_static_onnx_model(text, model, tokenizer, static_onnx_model_path)


# # step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
# if USE_FP16:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
# else:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch