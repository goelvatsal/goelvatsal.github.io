import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Cache model loading to avoid reloading on every interaction
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    device = 0 if torch.cuda.is_available() else -1
    text_gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return text_gen

text_gen = load_model()

st.title("Finance QA Bot")

# Input from user
user_q = st.text_input("Enter your finance question:")

if user_q:
    prompt = f"Explain the following finance concept in a detailed and informative way:\n\n{user_q}"
    with st.spinner("Generating answer..."):
        try:
            result = text_gen(
                prompt,
                max_new_tokens=250,
                min_new_tokens=25,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )[0]["generated_text"]
            st.markdown("### Answer:")
            st.write(result.strip())
        except Exception as e:
            st.error(f"Oops, something went wrong: {str(e)}")
