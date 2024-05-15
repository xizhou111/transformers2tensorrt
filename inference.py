import json
import numpy as np
from transformers import AutoTokenizer
import tensorrt as trt
import common
from tqdm import tqdm
from pprint import pprint
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

id2label = {0: "其他", 1: "语文", 2: "数学", 3: "英语", 4: "物理", 5: "化学", 6: "生物", 7: "历史", 8: "地理", 9: "政治"}
# 定义评估指标，包括准确率、精确率、召回率、F1值
def compute_metrics(predictions, labels):
    # logits, labels = eval_pred
    # predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions, output_dict=True, digits=4)
    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    # 添加每个类别的recall指标到结果字典
    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict):
            # results[f'recall_{label}'] = metrics['recall']
            try:
                results[f'recall_{id2label[int(label)]}'] = metrics['recall']
            except:
                results['label'] = metrics['recall']
    return results

"""
a、获取 engine，建立上下文
"""
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

engine_model_path = "./results/roberta_test.trt"
# Build a TensorRT engine.
engine = get_engine(engine_model_path)
# Contexts are used to perform inference.
context = engine.create_execution_context()

"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Load the evaluation dataset
eval_data_file = '/mnt/cfs/NLP/zcl/subjects_classification/fasttext/eval_data/eval_data.json'
with open(eval_data_file, 'r') as f:
    eval_data = json.load(f)

# eval_data = eval_data.rename(columns={"question": "text", "subject_id": "label"})
# eval_dataset = Dataset.from_pandas(eval_data)
tokenizer = AutoTokenizer.from_pretrained("/mnt/cfs/NLP/zcl/interface/http_demo/src/model/chinese-roberta-wwm-ext")

sentence = "如图,姐弟两人吃一个蛋糕。比较发现,(     )。\n姐姐：“我先吃这个蛋糕的 $$ \\frac{1}{3} $$。”\n弟弟：“我吃剩下的一半。”\nA. 姐姐吃得多  \nB. 弟弟吃得多  \nC. 两人吃得一样多\nD. 无法比较\n"
inputs = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True,max_length=256, padding="max_length", return_attention_mask=True,truncation=True)

tokens_id =  to_numpy(inputs['input_ids'].int())
attention_mask = to_numpy(inputs['attention_mask'].int())

context.active_optimization_profile = 0
origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
context.set_binding_shape(0, (origin_inputshape))               
context.set_binding_shape(1, (origin_inputshape))

"""
c、输入数据填充
"""
inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)

labels = []
predictions = []
for eval_batch in tqdm(eval_data):
    sentence = eval_batch['question']

    # sentence = "如图,姐弟两人吃一个蛋糕。比较发现,(     )。\n姐姐：“我先吃这个蛋糕的 $$ \\frac{1}{3} $$。”\n弟弟：“我吃剩下的一半。”\nA. 姐姐吃得多  \nB. 弟弟吃得多  \nC. 两人吃得一样多\nD. 无法比较\n"
    input = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True,max_length=256, padding="max_length", return_attention_mask=True,truncation=True)

    tokens_id =  to_numpy(input['input_ids'].int())
    attention_mask = to_numpy(input['attention_mask'].int())

    # context.active_optimization_profile = 0
    # origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
    # origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
    # context.set_binding_shape(0, (origin_inputshape))               
    # context.set_binding_shape(1, (origin_inputshape))

    # """
    # c、输入数据填充
    # """
    # inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
    inputs[0].host = tokens_id
    inputs[1].host = attention_mask

    """
    d、tensorrt推理
    """
    # time1 = time.time()
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    preds = np.argmax(trt_outputs, axis=1)

    labels.append(eval_batch['subject_id'])
    predictions.append(preds[0])




    

    # time2 = time.time()
    # print("====sentence=:", sentence)
    # print("====preds====:", preds)
    # print("====use time:=", time2-time1)


# 计算评估指标
results = compute_metrics(predictions, labels)
pprint(results)

