import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json

# Load model only once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

text_gen = load_model()

# Enable CORS manually
st.set_page_config(page_title="AI Backend", layout="centered")
st.title("Streamlit AI API")

# Parse incoming POST request from fetch()
if st.query_params.get("api") == "true":
    import sys
    import json

    try:
        body = st.runtime.scriptrunner.get_script_run_context().http_request.body
        if not body:
            st.error("No input body found.")
            st.stop()

        data = json.loads(body.decode())
        user_q = data.get("user_q", "")
        if not user_q:
            st.error("No question found.")
            st.stop()

        prompt = f"Explain the following finance concept in a detailed and informative way:\n\n{user_q}"
        result = text_gen(prompt, max_new_tokens=250, min_new_tokens=25, temperature=0.6, top_p=0.9, repetition_penalty=1.2, do_sample=True)
        st.json({"answer": result[0]["generated_text"].strip()})

    except Exception as e:
        st.json({"answer": f"Error: {str(e)}"})
    finally:
        st.stop()

# For dev testing
st.write("This backend expects POST requests at `?api=true` with JSON body `{ user_q: string }`.")
