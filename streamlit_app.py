import streamlit as st
import torch
from transformers import (
    BertTokenizer, BertForQuestionAnswering,
    BartTokenizer, BartForConditionalGeneration,
    pipeline
)
from datasets import load_dataset

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
def load_squad():
    st.info("Loading SQuAD dataset...")
    return load_dataset("squad", split="validation[:100]")

def load_cnn_dailymail():
    st.info("Loading CNN/DailyMail dataset...")
    return load_dataset("cnn_dailymail", "3.0.0", split="test[:50]")

squad_data = load_squad()
cnn_data = load_cnn_dailymail()

# Load Models
def load_models():
    st.info("Loading Transformer models...")
    qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(DEVICE)
    qa_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return qa_model, qa_tokenizer, summarizer

qa_model, qa_tokenizer, summarizer = load_models()

# Streamlit UI
st.title("🧠 NLP Toolkit: Question Answering, Summarization & Comprehension")
option = st.sidebar.radio("Choose a task:", ["Question Answering", "Text Summarization", "Document Comprehension"])

if option == "Question Answering":
    st.header("🔍 Question Answering")
    context = st.selectbox("Select a Context:", squad_data["context"][:10])
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True).to(DEVICE)
        outputs = qa_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
        )
        st.success(f"**Answer:** {answer}")

elif option == "Text Summarization":
    st.header("📜 Text Summarization")
    article = st.selectbox("Select an Article:", cnn_data["article"][:10])
    if st.button("Summarize"):
        summary = summarizer(article, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        st.success(f"**Summary:** {summary[0]['summary_text']}")

elif option == "Document Comprehension":
    st.header("📖 Document Comprehension")
    context = st.selectbox("Select a Document:", squad_data["context"][:10])
    question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
        response = qa_pipeline(question=question, context=context)
        st.success(f"**Answer:** {response['answer']}")
