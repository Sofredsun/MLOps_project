import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(page_title="Школьный ИИ-ассистент", layout="wide")

CHROMA_DIR = 'chroma_langchain_db'
AVAILABLE_MODELS = ['qwen2.5:7b', 'llama3.2']


@st.cache_resource
def load_knowledge_base():
    """
    Загружает векторную базу знаний. 
    Кешируется при первом запуске.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = Chroma(
        collection_name='school_knowledge_base',
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vector_store


# Шаблон промпта для языковой модели
TEMPLATE = """Вы — экспертный аналитик базы знаний школы. 
Ваша цель: найти ответ на вопрос в предоставленных фрагментах документов.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ИНСТРУКЦИЯ:
1. Проанализируй контекст. Если информация представлена в виде списка, таблицы или расписания — изучи каждую строку.
2. Если в тексте упоминаются похожие термины (например, "питание" вместо "завтрак"), используй их для ответа.
3. Если ответ найден частично, напиши то, что удалось найти.
4. Сначала кратко опиши, что ты нашел в документах, а затем дай итоговый ответ.

ОТВЕТ:"""


def main():
    st.title("Школьный ИИ-ассистент")
    st.markdown(
        "RAG-система для быстрого поиска информации по базе знаний школы: "
        "расписание, питание, правила, контакты и многое другое."
    )
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        
        # Выбор языковой модели
        selected_model = st.selectbox(
            "Языковая модель",
            options=AVAILABLE_MODELS,
            index=0
        )
        st.info(f"Модель: {selected_model}")
        st.info("База: ChromaDB (multilingual-e5-small)")
        
        # Количество документов из базы при поиске
        k_retrieval = st.slider("Документов для поиска (K)", 1, 12, 8)

        st.divider()
        if st.button("Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Задайте вопрос по базе знаний школы...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Ищу информацию в базе знаний..."):
                try:
                    # Загружаем базу знаний и инициализируем модель
                    vector_store = load_knowledge_base()
                    model = OllamaLLM(model=selected_model, temperature=0.1)
                    
                    # Настраиваем retriever для поиска похожих документов
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k_retrieval}
                    )

                    prompt = ChatPromptTemplate.from_template(TEMPLATE)
                    chain = prompt | model
                    
                    # Извлекаем релевантные документы и формируем контекст
                    docs = retriever.invoke(question)
                    context_text = "\n\n".join([
                        f"[Источник: {d.metadata.get('source', 'Неизвестно')}]\n{d.page_content}"
                        for d in docs
                    ])
                    
                    # Ответ от модели
                    response = chain.invoke({"context": context_text, "question": question})
                    
                    # Вывод ответа с эффектом печатания
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Сохраняем ответ ассистента в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.info("Убедитесь, что Ollama запущена и ChromaDB находится по пути chroma_langchain_db")


if __name__ == "__main__":
    main()
