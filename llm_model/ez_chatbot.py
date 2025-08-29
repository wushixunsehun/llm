import os, re
os.environ.pop("OPENAI_API_KEY",  None)
os.environ.pop("OPENAI_BASE_URL", None)
import sys, json, uuid, time
import signal
import logging
import subprocess
from mem0 import Memory
from openai import OpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import posthog
posthog.disabled = True


UPDATE_MEMORY_PROMPT = """你是智能记忆管理器，负责控制系统的记忆。
你可以执行四种操作：(1) 新增记忆 (ADD)、(2) 更新记忆 (UPDATE)、(3) 删除记忆 (DELETE)、(4) 不做改动 (NONE)。

根据上述四种操作，记忆内容将随之变化。

请将新检索到的事实与已有记忆进行比较。对每条新事实，决定是否：
- ADD：作为新的元素写入记忆
- UPDATE：用新事实更新已有记忆元素
- DELETE：删除已有记忆元素
- NONE：无需变动（事实已存在或与记忆无关）

具体操作指引如下：

1. **ADD（新增）**：若新检索到的事实包含记忆中未出现的新信息，则必须新增，并在 `id` 字段生成一个全新的 ID。
   **示例**
    - 旧记忆：
        [
            {"id": "0", "text": "用户是一名软件工程师"}
        ]
    - 新事实：["姓名是 John"]
    - 新记忆：
        {
            "memory": [
                {"id": "0", "text": "用户是一名软件工程师", "event": "NONE"},
                {"id": "1", "text": "姓名是 John", "event": "ADD"}
            ]
        }

2. **UPDATE（更新）**
    - 如果新事实与记忆中信息相同**但内容完全不同**，应执行更新。
    - 如果检索到的事实包含的信息与记忆中存在的元素传达的信息相同，则必须保留信息量最大的事实。
        例 (a) -- 如果内存中包含 "用户喜欢玩板球"，而检索到的事实是 "喜欢和朋友一起玩板球"，那么就用检索到的事实更新内存。
        例 (b) -- 如果记忆包含 "喜欢芝士披萨"，而检索到的事实是 "喜欢芝士披萨"，那么就不需要更新记忆，因为它们传达的是相同的信息。
    - 如果方向是更新记忆，那么就必须更新。
    - 更新时必须保持原 ID 不变，输出中的 ID 只能来自输入，**不得新生成 ID**。
   **示例**
    - 旧记忆：
        [
            {"id": "0", "text": "我非常喜欢奶酪披萨"},
            {"id": "1", "text": "用户是一名软件工程师"},
            {"id": "2", "text": "用户喜欢打板球"}
        ]
    - 新事实：["喜欢鸡肉披萨", "喜欢和朋友一起打板球"]
    - 新记忆：
        {
            "memory": [
                {
                    "id": "0",
                    "text": "喜欢奶酪和鸡肉披萨",
                    "event": "UPDATE",
                    "old_memory": "我非常喜欢奶酪披萨"
                },
                {
                    "id": "1",
                    "text": "用户是一名软件工程师",
                    "event": "NONE"
                },
                {
                    "id": "2",
                    "text": "喜欢和朋友一起打板球",
                    "event": "UPDATE",
                    "old_memory": "用户喜欢打板球"
                }
            ]
        }

3. **DELETE（删除）**
    - 若新事实与记忆信息相矛盾，则应删除。
    - 或者明确指令要求删除时，执行删除。
    - 删除时同样只能引用已有 ID，**不得新生成 ID**。
   **示例**
    - 旧记忆：
        [
            {"id": "0", "text": "姓名是 John"},
            {"id": "1", "text": "喜欢奶酪披萨"}
        ]
    - 新事实：["不喜欢奶酪披萨"]
    - 新记忆：
        {
            "memory": [
                {"id": "0", "text": "姓名是 John", "event": "NONE"},
                {"id": "1", "text": "喜欢奶酪披萨", "event": "DELETE"}
            ]
        }

4. **NONE（不变）**
    若新事实已包含在记忆中，或与记忆无关，无需任何改动。
   **示例**
    - 旧记忆：
        [
            {"id": "0", "text": "姓名是 John"},
            {"id": "1", "text": "喜欢奶酪披萨"}
        ]
    - 新事实：["姓名是 John"]
    - 新记忆：
        {
            "memory": [
                {"id": "0", "text": "姓名是 John", "event": "NONE"},
                {"id": "1", "text": "喜欢奶酪披萨", "event": "NONE"}
            ]
        }

**补充规则**
- **务必保持 `id` 的“类型”与“内容”完全一致**，例如不要把 `6` 写成 `"6"`。
- **若新旧事实语义一致（仅措辞或同义词变化），返回 `event = "NONE"`；只有信息冲突或补充细节时才使用 `UPDATE`。**
"""


llm_model = "Qwen/Qwen3-30B-A3B"
llm_host = "http://a6000-G5500-V6:5414/v1"
embedding_model = "Alibaba-NLP/gte-multilingual-base"
embedding_host = "http://a6000-G5500-V6:5415/v1"
api_key = "EMPTY"

milvus_collection = "mem0_test"
milvus_host = "http://a6000-G5500-V6:19530"
milvus_token = "root:lhltxh971012"


config = {
    "version": "v1.1",

    # --- 8< --- 本地 LLM 服务 --- 8< ---
    "llm": {
        "provider": "openai",
        "config": {
            "model": llm_model,
            "openai_base_url": llm_host,
            "api_key": api_key,
        }
    },

    # --- 8< --- 本地 Embedding 服务 --- 8< ---
    "embedder": {
        "provider": "openai",
        "config": {
            "model": embedding_model,
            "openai_base_url": embedding_host,
            "api_key": api_key,
            "embedding_dims": 768,
        }
    },

    # --- 8< --- Milvus 向量库 --- 8< ---
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": milvus_collection,
            "embedding_model_dims": 768,
            "url": milvus_host,
            "token": milvus_token,
        }
    },

    # --- 8< --- 自定义更新记忆提示词 --- 8< ---
    "prompts": {
        "update_memory": UPDATE_MEMORY_PROMPT,
    }
}


mem0 = Memory.from_config(config)
OPENAI_CLIENT = OpenAI(base_url=llm_host, api_key=api_key)


logger = logging.getLogger("ez_chatbot")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("backoff").handlers.clear()
logging.getLogger("backoff").propagate = False
logging.getLogger("backoff").addHandler(logging.NullHandler())
logging.getLogger("backoff").setLevel(logging.CRITICAL + 1)


if not logger.handlers:
    file_handler = logging.FileHandler("llm_model/ez_chatbot.log")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)


def concat(original: list, new: list) -> list:
    return original + new


class ChatState(TypedDict):
    user_id: str
    messages: Annotated[list, concat]


# def build_prompt_with_search(user_input, memory_context):
#     # 初始化 SearXNG 搜索包装器
#     search = SearxSearchWrapper(searx_host="http://a6000-G5500-V6:5415", k=5)

#     # 获取搜索结果
#     search_results = search.run(user_input)
    
#     # 构建提示词
#     prompt = PROMPT_TEMPLATE.format(
#         search_results = search_results,
#         memory_context = memory_context,
#         user_input = user_input
#     )

#     return prompt


def report_memory_changes(mem_results: list[dict]) -> None:
    """
    把 Mem0 返回的结果分组打印到终端和日志。
    只处理 event≠NONE 的条目；完全无变动时静默。
    """
    changes = {"ADD": [], "UPDATE": [], "DELETE": []}

    for r in mem_results:
        evt = r.get("event", "NONE")
        if evt != "NONE":
            changes[evt].append(r)

    if not any(changes.values()):
        return  # 没有任何新增 / 更新 / 删除

    lines: list[str] = []
    if changes["ADD"]:
        lines.append("🔹 新增记忆:")
        lines.extend([f'  + {m["memory"]}' for m in changes["ADD"]])

    if changes["UPDATE"]:
        lines.append("🔸 更新记忆:")
        lines.extend(
            [
                f'  ~ {m["old_memory"]}  →  {m["memory"]}'
                if "old_memory" in m
                else f'  ~ {m["memory"]}'
                for m in changes["UPDATE"]
            ]
        )

    if changes["DELETE"]:
        lines.append("🔻 删除记忆:")
        lines.extend([f'  - {m["memory"]}' for m in changes["DELETE"]])

    msg_out = "\n".join(lines)
    print(f'\n{msg_out}\n')
    logger.info("\n" + msg_out + "\n")


def strip_think(text: str) -> str:
    """删除 <think>…</think> 块"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def query_llm(state: ChatState, *, stream: bool = True, show_thought: bool = True) -> ChatState:
    """
    向 LLM 发送 user 消息；
    若 stream=True 则边生成边打印，同时在结束后补写记忆。
    """
    user_input = state["messages"][-1]
    user_id    = state["user_id"]

    # ---记忆检索 ---
    mem_hits = mem0.search(user_input, user_id=user_id, limit=3)
    mem_ctx = "\n".join(f"- {r['memory']}" for r in mem_hits["results"])
    system_msg = (
        "你是一名智能助理。\n以下是与用户相关的记忆：\n"
        f"{mem_ctx}\n请回答用户提问："
    )

    # --- 调用 LLM ---
    completion = OPENAI_CLIENT.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_input},
        ],
        stream=stream,
        extra_body={"enable_thinking": show_thought},
    )

    visible_parts: list[str] = []
    clean_parts:   list[str] = []

    if stream:
        for chunk in completion:
            delta_raw = chunk.choices[0].delta.content or ""
            delta_vis = delta_raw if show_thought else strip_think(delta_raw)
            if delta_vis:
                print(delta_vis, end="", flush=True)
            visible_parts.append(delta_vis)
            clean_parts.append(strip_think(delta_raw))
        print()   # 换行
    else:
        full_raw = completion.choices[0].message.content
        visible_parts.append(full_raw if show_thought else strip_think(full_raw))
        clean_parts.append(strip_think(full_raw))

    full_visible = "".join(visible_parts)
    full_clean   = "".join(clean_parts)

    # --- 写入记忆 ---
    mem_msg = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": full_clean},
    ]

    try:
        mem_result = mem0.add(
            mem_msg,
            user_id=user_id,
            metadata={"source": "chat", "time": time.time()},
        )
        report_memory_changes(mem_result.get("results", []))
    except json.JSONDecodeError as e:
        logging.warning(f"[mem0] add failed, skip: {e}")

    # --- 更新状态 State ---
    state["messages"].append(full_visible)
    return state


def build_graph() -> StateGraph:
    workflow = StateGraph(ChatState)
    workflow.add_node(query_llm)
    workflow.set_entry_point("query_llm")
    workflow.add_edge("query_llm", "__end__")
    return workflow.compile()


def check_systemd_service(service_name: str):
    try:
        result = subprocess.run([
            'systemctl', 'is-active', service_name
        ], capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except Exception as e:
        print(f"检查 systemd 服务时出错: {e}")
        return False


def check_dependencies() -> None:
    svc = "qwq30b.service"
    if not check_systemd_service(svc):
        sys.exit(f"❌ 依赖的 systemd 服务 {svc} 未启动，请先启动后再运行本程序！")


# --- 8< --- 单例初始化 --- 8< ---
GRAPH = build_graph()
THREAD_ID = str(uuid.uuid4())


def main() -> None:
    # check_dependencies()

    # 仅截 SIGTERM；Ctrl-C 维持默认
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit("\n👋 收到 SIGTERM，已退出！"))

    login = input("新会话，请输入用户名：").strip() or "anonymous"
    print(f"🤖 欢迎！ {login}，现在就开始聊天吧！")

    state = ChatState(
        user_id=login,
        messages=[]
    )

    state = ChatState(user_id=login, messages=[])
    thread_cfg = {"configurable": {"thread_id": THREAD_ID}}

    chat_cnt = 0

    # 3) 主聊天循环
    try:
        while True:
            user_input = input(f"\n[{chat_cnt}] >>> ")
            if not user_input.strip():
                print("❌ 输入不能为空，请重新输入。")
                continue

            logger.info(f"\n[{chat_cnt}] >>> {user_input}")

            result = GRAPH.invoke(
                {"user_id": state["user_id"], "messages": [user_input]},
                config=thread_cfg
            )

            if not result.get("streamed", False):
                reply = result["messages"][-1]
                # print(f"[{chat_cnt}] AI: {reply}")
                logger.info(f"[{chat_cnt}] AI: {reply}")

            chat_cnt += 1
    except KeyboardInterrupt:
        # 捕获 Ctrl-C 退出
        print("\n👋 已退出，再见！")


if __name__ == "__main__":
    main()
