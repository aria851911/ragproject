import os
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app_rag_prod import (
    EMB_MODEL,
    load_index_and_chunks,
    process_query,
)


# =========================================================
# 路徑與環境變數
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent

ENV_CANDIDATES = [
    CURRENT_DIR / ".env",
    CURRENT_DIR.parent / ".env",
]


def load_env_file() -> None:
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    load_dotenv(override=False)


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY。請確認 .env 檔存在，且裡面有 OPENAI_API_KEY=你的金鑰"
        )
    return api_key


# =========================================================
# 初始化（只跑一次）
# =========================================================

def init_system() -> Tuple[OpenAI, SentenceTransformer, Any, Any]:
    load_env_file()

    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    emb_model = SentenceTransformer(EMB_MODEL)
    index, chunks = load_index_and_chunks()

    return client, emb_model, index, chunks


# =========================================================
# 單次問答（給 UI 呼叫）
# =========================================================

def chat_once(
    user_input: str,
    client: OpenAI,
    emb_model: SentenceTransformer,
    index: Any,
    chunks: Any,
) -> Dict[str, Any]:
    if not isinstance(user_input, str) or not user_input.strip():
        return {
            "answer": "請先輸入內容。",
            "mode": "invalid_input",
            "raw": {"error": "empty_input"},
        }

    result = process_query(
        client=client,
        emb_model=emb_model,
        index=index,
        chunks=chunks,
        user_input=user_input.strip(),
    )

    return {
        "answer": result.get("l5_answer", ""),
        "mode": result.get("mode", ""),
        "raw": result,
    }