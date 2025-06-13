import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import requests
import asyncio
import json
from difflib import get_close_matches

st.set_page_config(page_title="AI Backend", layout="centered")

# Ensure event loop works on all platforms
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Download JSON dataset if not present
json_path = "finance_questions_formatted.json"
if not os.path.exists(json_path):
    response = requests.get("https://github.com/goelvatsal/goelvatsal.github.io/releases/download/dataset/finance_questions_formatted.json")
    with open(json_path, "wb") as f:
        f.write(response.content)

# Load the JSON cache
with open(json_path, "r", encoding="utf-8") as f:
    json_cache = json.load(f)

# Search JSON cache for close match
def get_cached_answer(query):
    questions = [item["instruction"] for item in json_cache]
    matches = get_close_matches(query.lower(), [q.lower() for q in questions], n=1, cutoff=0.85)
    if matches:
        matched_question = next(q for q in questions if q.lower() == matches[0])
        for item in json_cache:
            if item["instruction"] == matched_question:
                return item["output"]
    return None

# Load model pipeline
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

text_gen = load_model()

# UI and state
st.title("AI Finance Chatbot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_q = st.text_area("Enter your finance question here:")

if st.button("Get Explanation"):
    if not user_q.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Checking cached answers..."):
            cached = get_cached_answer(user_q)

        if cached:
            answer = cached.strip()
            st.success("Answer loaded from cache")
        else:
            with st.spinner("Generating answer using the model..."):
                prompt = (
                    f"You are a financial analyst that's designed to help answer broad and general financial questions in a descriptive and helpful way. "
                    f"Explain the following finance concept in a detailed and informative way and answer every part of this question and provide some examples:\n\n{user_q}"
                )
                result = text_gen(
                    prompt,
                    max_new_tokens=300,
                    min_new_tokens=40,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.6,
                    do_sample=False,
                )
                answer = result[0]["generated_text"].strip()

        st.session_state.chat_history.append({"instruction": user_q.strip(), "output": answer})

# Show conversation history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** \"{entry['instruction']}\"")
        st.markdown(f"**A{i+1}:** {entry['output']}")
        st.markdown("")
