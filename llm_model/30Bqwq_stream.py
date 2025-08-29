import os
# export VLLM_USE_MODELSCOPE=true
os.environ['VLLM_USE_MODELSCOPE'] = 'true'    # 自动下载模型时，指定使用 modelscope。不设置的话，会从 huggingface 下载
# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'    # 设置显卡设备
os.environ['VLLM_USE_V1'] = '1'    # 启用分块预填充


import json
import uuid
import time
import yaml
import torch
import uvicorn
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing import AsyncGenerator


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(current_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

llm_conf: dict = config['llm']
conf_max_tokens = llm_conf.get("max_tokens")    # 最大生成长度
conf_temperature = llm_conf.get("temperature")    # 温度，控制生成文本的多样性
conf_top_k = llm_conf.get("top_k")    # 核采样的候选词数量
conf_top_p = llm_conf.get("top_p")    # 核采样的概率
conf_presence_penalty = llm_conf.get("presence_penalty")    # 存在惩罚，控制生成文本的多样性


# 指定模型名称，自动下载模型
# qwqmodel = 'tclf90/qwq-32b-gptq-int4'
# qwqmodel = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# qwqmodel = 'okwinds/QwQ-32B-Int8-W8A16'
qwqmodel = llm_conf.get("model")
# qwqmodel="qwen/Qwen2-7B-Instruct"


# 加载分词器，用于将输入文本转换为模型可处理的格式
tokenizer = AutoTokenizer.from_pretrained(qwqmodel)
# print(torch.cuda.current_device())


qwq = LLM(
        model = qwqmodel,
        tokenizer = qwqmodel,
        trust_remote_code = True,
        max_model_len = conf_max_tokens,
        tensor_parallel_size = 2,    # 张量并行 (Tensor Parallel, TP)
        pipeline_parallel_size = 1,    # 流水线并行 (Pipeline Parallel, PP)
        gpu_memory_utilization = 0.9,    # 显存利用率
        block_size = 16,
        max_num_batched_tokens = 8192,    # 配合 chunk prefill，提高吞吐
        max_num_seqs = 384,    # 并发序列数调高一些，适配 fastapi 多请求
        disable_log_stats = True,
        enforce_eager = True,    # 加快启动速度, 允许 CUDA graph
    )


# 创建简易服务器
app = FastAPI()


# 允许外域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_completion_stream(llm: LLM, prompt,
                        max_tokens = conf_max_tokens,
                        temperature = conf_temperature,
                        top_k = conf_top_k,
                        top_p = conf_top_p,
                        presence_penalty = conf_presence_penalty):
    """同步生成器：每个 chunk 是一个增量输出"""
    stop_token_ids = []

    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        max_tokens = max_tokens,
        stop_token_ids = stop_token_ids,
        presence_penalty = presence_penalty
    )

    # 开启流式生成
    for output in llm.generate(prompt, sampling_params, stream=True):
        yield output.outputs[0].text


@app.post("/v1/chat/completions")
async def run_stream(request: Request) -> StreamingResponse:
    data = await request.json()
    messages = data["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False
    )

    async def stream_response():
        try:
            for chunk in get_completion_stream(qwq, [text]):
                if chunk.strip():  # 确保分块非空
                    yield chunk
        except Exception as e:
            yield f"Error: {str(e)}\n"

    return StreamingResponse(stream_response(), media_type="text/plain")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=5414, workers=1)


if __name__ == '__main__':
    main()
