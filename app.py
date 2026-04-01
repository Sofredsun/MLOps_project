import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(page_title="Школьный ИИ-ассистент", layout="wide")

DATA_DIR = 'data/school_knowledge_base'
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
        
        k_retrieval = st.slider("Количество документов для поиска", 1, 12, 8)

        st.divider()
        
        # Список готовых вопросов для быстрого старта
        st.subheader("Примеры вопросов")
        example_questions = [
            "Во сколько начинается первый урок?",
            "Когда весенние каникулы?",
            "Зачем нужна внеурочная деятельность?",
            "Как связаться с директором?",
            "Что на завтрак в 5 классе?",
            "Какие правила поведения для родителей?",
        ]
        
        # При клике на кнопку сохраняем вопрос в session_state, 
        # он будет подставлен в поле ввода как будто пользователь написал его сам
        for eq in example_questions:
            if st.button(eq, use_container_width=True):
                st.session_state["prefill_question"] = eq
                
        st.divider()
        if st.button("Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Если в сообщении ассистента есть источники, то показываем их в раскрывающемся блоке
            if "sources" in message and message["sources"]:
                with st.expander("Источники"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**{idx + 1}. {source['source']}**")
                        st.caption(source['content'][:300] + "...")

    prefill = st.session_state.pop("prefill_question", None)

    question = st.chat_input("Задайте вопрос по базе знаний школы...")

    if prefill and not question:
        question = prefill

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
                    
                    sources_data = [
                        {
                            "source": doc.metadata.get("source", "Неизвестно"),
                            "content": doc.page_content
                        }
                        for doc in docs
                    ]
                    
                    # Сохраняем ответ ассистента в историю
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_data
                    })
                    
                    with st.expander("Найденные фрагменты документов"):
                        for idx, doc in enumerate(docs):
                            st.markdown(f"**Фрагмент {idx + 1}:** `{doc.metadata.get('source', 'Неизвестно')}`")
                            st.caption(doc.page_content[:400] + "...")
                            if idx < len(docs) - 1:
                                st.divider()

                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.info("Убедитесь, что Ollama запущена и ChromaDB находится по пути chroma_langchain_db")


if __name__ == "__main__":
    main()
