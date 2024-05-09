import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "/mnt/cfs/NLP/zcl/interface/http_demo/src/model/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

torch.save(model.state_dict(), "./results/roberta_params.pth")
print("模型保存成功")