###
# m-llama3-ntp.py
# m = MPI数据集，论文是NIPS23那篇，很简单的论文，不妨看一看
# ntp = next token prediction, 即希望模型通过预测下一个token是什么的方式来做出对于MPI数据集中问题的答案
###

MODEL_PATH = "./Llama-3-8B-Instruct" # 这个模型我记得你好像用过吧，没有的话就去huggingface下载，记得放在当前目录下
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main 不需要下载original文件夹里的东西
# 这个模型不管用的话，你也可以试一试Llama-3-8B，就是没有经过instruct版本的

ITEMPATH = "./mpi_120.csv" # MPI数据集，你看一眼就知道是咋回事了，它第一列那个label_raw的内容不重要，第二列即人格描述
# MPI论文提出的做法是，模型需要通过对这个描述进行打分，然后每个描述对应五个人格维度的一种，最后一列 1 or -1 表示是正向还是负向的人格


template = """Someone thinks of me as: 'You {}.' However, if I had to decide right now, I would say this statement is most likely to be"""
#!!!!!!!这个template就是你需要自己不断修改尝试的地方!!!!!!!
# 我这里用的这个就是希望它能接着这个template来回复一个词出来，然后这个词使得我们去对应到模型对于这个描述的态度
# 以第一个MPI举例：原始的MPI数据（mpi_120.csv的第一行）：Anxiety,Worry about things,N,1
# 我们给大模型输入 Someone thinks of me as: 'You Worry about things.' However, if I had to decide right now, I would say this statement is most likely to be
# 我们希望它接着输出一个词（next token prediction）类似 correct,incorrect 或者 true, false 或者 right, wrong等等，你要想一想还可以是哪些词
# 这样就虽然不是像MPI原论文那样是一个打分的形式，但也可以通过“这个词”来进行一个打分
# 比如它回复了true，那我们就可以说他在N这个人格维度上是比较高的

messages = [
    {"role":"system","content":"You should only answer true or false."},# !!!!!!这也是你要尝试修改的地方!!!!!!
    # llama模型的输入包含一个system部分，相当于你给它定了一个全局的“人设”
    # 比如我在这里希望它就回复true or false
    {"role":"user","content":template},
]
# 现在这个template就不行，你试一下就知道，他就只能回复false这个词。

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
import numpy as np
import transformers
from copy import deepcopy

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
global_result = {}
global_cnt = {}


def getItems(filename=ITEMPATH, item_type=None):
    data = pd.read_csv(filename)
    if item_type is not None:
        items = data[data["instrument"] == item_type]
    else:
        items = data
    return items

def getPipeline():
    pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={"torch_dtype": torch.float16},
    device=device,
)
    return pipeline


def generateAnswer(pipeline, dataset, messages):
    global_result = []
    for _, item in dataset.iterrows():
        question = item["text"].lower()
        input_messages = deepcopy(messages)
        input_messages[1]["content"] = messages[1]["content"].format(question)
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            input_messages,
            max_new_tokens=20,
            eos_token_id=terminators,
            pad_token_id = pipeline.tokenizer.eos_token_id,
            temperature=0.0,
            top_p=0.95,
        )
        output_text = outputs[0]["generated_text"]
        print("-" * 40)
        print(output_text)
        global_result.append(output_text)


    return global_result


def main():
    pipeline = getPipeline()
    dataset = getItems(ITEMPATH,None)
    print("-" * 40)
    print(f"Current Prompt: {messages}")

    result = generateAnswer(pipeline, dataset, messages)
    np.save("./result.npy",result)



if __name__ == "__main__":
    main()
