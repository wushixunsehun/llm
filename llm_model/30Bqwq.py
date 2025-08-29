import os, uuid, time, yaml, json
os.environ["VLLM_USE_MODELSCOPE"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_V1"] = "0"         # 与 V0 Engine 保持一致


import uvicorn
from vllm import SamplingParams
from fastapi import FastAPI, Request
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi.middleware.cors import CORSMiddleware
from vllm.engine.async_llm_engine import AsyncLLMEngine  # V0 Engine
from fastapi.responses import StreamingResponse, JSONResponse


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(current_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)["llm"]


# 指定模型名称和分词器
qwqmodel = cfg["model"]
tokenizer = AutoTokenizer.from_pretrained(qwqmodel)


args = AsyncEngineArgs(
    model=qwqmodel,
    tokenizer=qwqmodel,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=cfg["max_tokens"],
    block_size=16,
    max_num_batched_tokens=32768,
    max_num_seqs=384,
    trust_remote_code=True,
    disable_log_stats=True,      # 关闭 throughput 等统计
    disable_log_requests=True,   # 关闭 Added/Finished request
)
engine = AsyncLLMEngine.from_engine_args(args)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    want_stream = bool(data.get("stream", False))    # 是否流式输出，默认否
    enable_thinking = bool(data.get("enable_thinking", False))    # 是否启用模型的思考能力
    messages = data["messages"]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    sampling = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        max_tokens=cfg["max_tokens"],
        presence_penalty=cfg["presence_penalty"],
    )

    req_id = str(uuid.uuid4())
    created = int(time.time())
    gen = engine.generate(prompt, sampling, req_id)    # 异步生成器

    # ============ 流式 ============ #
    if want_stream:
        async def event_stream():
            sent   = ""    # 已推文本
            async for out in gen:
                full = out.outputs[0].text
                delta = full[len(sent):]    # 只取新增
                sent  = full
                if delta:
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": cfg["model"],
                        "choices": [{
                            "index": 0,
                            "delta": { "content": delta },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # 结束标记
            yield "data: [DONE]\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    # ---------- B. 非流式 ----------
    async for out in gen:    # 仅为取最终结果
        final = out
    text = final.outputs[0].text

    return JSONResponse({
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5414)
