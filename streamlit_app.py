import streamlit as st
import torch
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline,
)
import pandas as pd
from datasets import load_dataset

def load_squad():
    st.info("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:100]")
    return pd.DataFrame(dataset)

def load_cnn_dailymail():
    st.info("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:50]")
    return pd.DataFrame(dataset)

def question_answering(squad_df):
    st.subheader("Question Answering Task")
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    
    context_options = squad_df["context"].unique()
    selected_context = st.selectbox("Choose a context:", context_options[:10])
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        inputs = tokenizer(question, selected_context, return_tensors="pt", truncation=True).to(model.device)
        outputs = model(**inputs**)
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
        )
        st.success(f"Answer: {answer}")

def text_summarization(cnn_df):
    st.subheader("Text Summarization Task")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    
    article = st.selectbox("Choose an article:", cnn_df["article"].head(10))
    if st.button("Summarize"):
        inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success(f"Summary: {summary}")

def document_comprehension(squad_df):
    st.subheader("Document Comprehension Task")
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    context_options = squad_df["context"].unique()
    selected_context = st.selectbox("Choose a context:", context_options[:10])
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        response = nlp(question=question, context=selected_context)
        st.success(f"Answer: {response['answer']}")

def main():
    st.title("Transformer-based NLP Toolkit")
    squad_df = load_squad()
    cnn_df = load_cnn_dailymail()
    
    option = st.sidebar.selectbox("Select a Task", ["Question Answering", "Text Summarization", "Document Comprehension"])
    
    if option == "Question Answering":
        question_answering(squad_df)
    elif option == "Text Summarization":
        text_summarization(cnn_df)
    elif option == "Document Comprehension":
        document_comprehension(squad_df)

if __name__ == "__main__":
main()
