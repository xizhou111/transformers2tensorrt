# transformers模型保存为pth文件
import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

if __name__ == '__main__':

    checkpoint = "/mnt/cfs/NLP/zcl/interface/http_demo/src/model/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    torch.save(model.state_dict(), "./results/roberta_params.pth")
    print("transformers模型保存pth模型成功！")