# transformers模型保存为pth文件
import torch
from transformers import  AutoTokenizer
from transformers import AutoModelForSequenceClassification

if __name__ == '__main__':

    checkpoint = "/mnt/cfs/NLP/zcl/subjects_classification/roberta/output_pretrain_512/checkpoint-30000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    torch.save(model.state_dict(), "./results/roberta_pretrain_512.pth")
    print("transformers模型保存pth模型成功！")