import os
# 可选：设置环境变量确保使用本地缓存
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HOME'] = '/home/tanxh/.cache/huggingface'  # 明确指定缓存路径


import yaml
import torch
import uvicorn
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(current_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# 配置模型路径
MODEL_NAME = config['rag']['embedding_model']
app = FastAPI(title="OpenAI Compatible Embedding API")


def get_local_model_path(model_name: str) -> str:
    """
    将 'Alibaba-NLP/gte-multilingual-base' 转换成本地 cache 路径
    如 ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-multilingual-base/snapshots/<hash>
    """
    cache_dir = '/home/sehun/.cache/huggingface/hub'
    cache_dir = Path(cache_dir).resolve()

    safe_name = model_name.replace("/", "--")
    model_dir = cache_dir / f"models--{safe_name}" / "snapshots"
    
    # 默认只取第一个 snapshot 子目录（一般只有一个）
    snapshot_dirs = list(model_dir.glob("*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for model: {model_name}")
    
    return str(snapshot_dirs[0])


# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model_path = get_local_model_path(MODEL_NAME)
model = SentenceTransformer(
    embedding_model_path,
    device = device,
    trust_remote_code = True,
    local_files_only = True,
)


class OpenAIEmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "gte-multilingual-base"
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def create_embedding(request: OpenAIEmbeddingRequest):
    try:
        inputs = [request.input] if isinstance(request.input, str) else request.input
        
        # 生成嵌入向量
        embeddings = model.encode(
            inputs,
            convert_to_tensor = True,
            normalize_embeddings = True,
            show_progress_bar = False
        )
        
        # 转换为列表格式
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # 构建响应
        response_data = []
        for i, embedding in enumerate(embeddings):
            response_data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
        
        # print(response_data)
        return {
            "object": "list",
            "data": response_data,
            "model": request.model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in inputs),
                "total_tokens": sum(len(text.split()) for text in inputs)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    uvicorn.run("embedding_api:app", host="0.0.0.0", port=5415)


if __name__ == '__main__':
    main()
