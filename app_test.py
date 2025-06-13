import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from intel_extension_for_transformers.transformers.pipeline import pipeline
import asyncio

st.set_page_config(page_title="AI Backend", layout="centered")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

text_gen = load_model()

st.title("AI Finance Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_q = st.text_area("Enter your finance question here:")

if st.button("Get Explanation"):
    if not user_q.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Generating answer..."):
            prompt = (f"You are a financial analyist that's designed to help answer broad and general financial questions in a descriptive and helpful way. "
                      f"Explain the following finance concept in a detailed and informative way and answer every part of this question and provide some examples:\n\n{user_q}")
            result = text_gen(
                prompt,
                max_new_tokens=300,
                min_new_tokens=40,
                temperature=0.4, #lower makes it focused, higher makes it random
                top_p=0.9,
                repetition_penalty=1.6,
                do_sample=False,
            )
            answer = result[0]["generated_text"].strip()
            st.session_state.chat_history.append({"question": user_q.strip(), "answer": answer})

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** \"{entry['question']}\"")
        st.markdown(f"**A{i+1}:** {entry['answer']}")
        st.markdown("")
