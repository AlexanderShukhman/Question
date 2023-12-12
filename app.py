import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
   return pipeline("question-answering", model="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")

qa_model = load_model()
st.title("📝 Ответы на вопросы по тексту")
uploaded_file = st.file_uploader("Загрузите текст", type=("txt", "md"))
question = st.text_input(
    "Введите вопрос по тексту",
    placeholder="Мой вопрос?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()
    results = qa_model(question = question, context = article)

    st.write("Ответ")
    st.write(results["answer"])
