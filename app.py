import streamlit as st
from core_engine import init_system, chat_once


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="人格互動顧問",
    page_icon="🧠",
    layout="centered",
)


# =========================================================
# 初始化系統（只跑一次）
# =========================================================

@st.cache_resource
def load_system():
    return init_system()


client, emb_model, rag_stores = load_system()


# =========================================================
# Chat session
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def add_user(msg: str):
    st.session_state.messages.append({
        "role": "user",
        "content": msg,
    })


def add_ai(msg: str, mode: str):
    st.session_state.messages.append({
        "role": "assistant",
        "content": msg,
        "mode": mode,
    })


# =========================================================
# Helper
# =========================================================

def render_mode_badge(mode: str) -> None:
    mode_map = {
        "marketing_copy": "### 📣 行銷文案",
        "strategy_advice": "### 🧠 人格洞察",
        "interpersonal_chat": "### 🤝 人際分析",
        "other": "### 💬 回覆",
        "error": "### ⚠️ 系統提示",
        "invalid_input": "### ⚠️ 輸入提醒",
    }

    label = mode_map.get(mode, "")
    if label:
        st.markdown(label)


# =========================================================
# UI
# =========================================================

st.title("🧠 人格互動顧問")


# 顯示歷史訊息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_mode_badge(m.get("mode", ""))
            st.markdown(m["content"])
        else:
            st.markdown(m["content"])


# 輸入框
user_input = st.chat_input("請輸入你的問題")


# =========================================================
# 主流程
# =========================================================

if user_input and user_input.strip():
    text = user_input.strip()

    add_user(text)

    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = chat_once(
                    user_input=text,
                    client=client,
                    emb_model=emb_model,
                    rag_stores=rag_stores,
                )

                answer = result.get("answer", "").strip()
                mode = result.get("mode", "").strip()

                render_mode_badge(mode)
                st.markdown(answer)
                add_ai(answer, mode)

            except Exception as e:
                err = f"系統目前發生問題：{e}"
                render_mode_badge("error")
                st.warning(err)
                add_ai(err, "error")