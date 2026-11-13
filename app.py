"""
TunedIn chat UI.

    streamlit run app.py
"""

import streamlit as st

from src.chatbot import TunedInBot

st.set_page_config(page_title="TunedIn", page_icon="🎵")
st.title("🎵 TunedIn")
st.caption("Tell me how you're feeling and I'll find songs that fit.")


@st.cache_resource
def get_bot():
    return TunedInBot()


bot = get_bot()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

if message := st.chat_input("How are you feeling?"):
    st.session_state.history.append(("user", message))
    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        with st.spinner("Finding songs..."):
            reply = bot.answer(message)
        text = reply.text.replace("\n", "  \n")  # keep line breaks in markdown
        st.markdown(text)
        with st.expander("Why these songs?"):
            st.write(f"Mood read as **{reply.emotion}** by the {reply.mood_source} model; "
                     f"reply from **{reply.reply_source}** in {reply.latency_s:.1f}s.")
            if reply.profile:
                st.json(reply.profile)
            for e in reply.errors:
                st.warning(e)
    st.session_state.history.append(("assistant", text))
