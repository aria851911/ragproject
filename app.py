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


client, emb_model, index, chunks = load_system()


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
# UI
# =========================================================

st.title("🧠 人格互動顧問")


# 顯示歷史訊息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# 輸入框
user_input = st.chat_input("請輸入你的問題")


# =========================================================
# 主流程
# =========================================================

if user_input and user_input.strip():
    text = user_input.strip()

    add_user(text)

    with st.chat_message("user"):
        st.write(text)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = chat_once(
                    user_input=text,
                    client=client,
                    emb_model=emb_model,
                    index=index,
                    chunks=chunks,
                )

                answer = result.get("answer", "")
                mode = result.get("mode", "")

                st.write(answer)
                add_ai(answer, mode)

            except Exception as e:
                err = f"系統錯誤：{e}"
                st.error(err)
                add_ai(err, "error")