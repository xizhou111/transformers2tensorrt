from transformers import pipeline
import time


checkpoint = "/mnt/cfs/NLP/zcl/interface/http_demo/src/model/chinese-roberta-wwm-ext"

classifier = pipeline('text-classification', model=checkpoint, tokenizer=checkpoint)

if __name__ == '__main__':
    for i in range(20):
        start = time.time()

        result = classifier("如图,姐弟两人吃一个蛋糕。比较发现,(     )。\n姐姐：“我先吃这个蛋糕的 $$ \\frac{1}{3} $$。”\n弟弟：“我吃剩下的一半。”\nA. 姐姐吃得多  \nB. 弟弟吃得多  \nC. 两人吃得一样多\nD. 无法比较\n")

        end = time.time()
        print(result, end-start)
