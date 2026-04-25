import time

import requests
import streamlit as st

st.set_page_config(page_title="Школьный ИИ-ассистент", layout="wide")

CHROMA_DIR = "chroma_langchain_db"
AVAILABLE_MODELS = ["qwen2.5:7b", "llama3.2"]
API_URL = "http://localhost:8000"


@st.cache_data(ttl=30)
def fetch_alerts():
    """Загружает алерты с бэкенда каждые 30 секунд"""
    try:
        response = requests.get(f"{API_URL}/monitoring/alerts", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def main():
    # Блок отображения уведомлений о дрейфе
    alerts = fetch_alerts()
    if alerts:
        latest_alert = alerts[0]
        if latest_alert.get("drift_detected"):
            score = latest_alert.get("drift_score", "N/A")
            threshold = latest_alert.get("threshold", "N/A")
            recommendation = latest_alert.get("recommendation", "Проверьте данные")
            timestamp = latest_alert.get("timestamp", "N/A")

            st.error(f"**Обнаружен дрейф запросов!**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Drift Score", f"{score}")
                st.metric("Threshold", f"{threshold}")
            with col2:
                st.metric("Time", timestamp[:16] if timestamp != "N/A" else "N/A")
                st.info(f"{recommendation}")

            if st.button("Обновить статус дрейфа"):
                requests.get(f"{API_URL}/monitoring/drift", timeout=10)
                st.cache_data.clear()
                st.rerun()
            st.divider()

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
            "Языковая модель", options=AVAILABLE_MODELS, index=0
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
            # Если в сообщении ассистента есть источники,
            # то показываем их в раскрывающемся блоке
            if "sources" in message and message["sources"]:
                with st.expander("Источники"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**{idx + 1}.{source['source']}**")  # type: ignore
                        st.caption(source["content"][:300] + "...")  # type: ignore

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

            with st.spinner("Ищу информацию в базе знаний..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "model": selected_model,
                            "k_retrieval": k_retrieval,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    data = response.json()

                    full_response = data["answer"]
                    sources_data = data["sources"]
                    latency = data["latency"]

                    # Эффект печатания
                    displayed = ""
                    for chunk in full_response.split():
                        displayed += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(displayed + "▌")
                    message_placeholder.markdown(displayed)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": displayed,
                        "sources": sources_data,
                    })

                    with st.expander("Найденные фрагменты документов"):
                        for idx, source in enumerate(sources_data):
                            st.markdown(f"**Фрагмент {idx + 1}:** `{source['source']}`")
                            st.caption(source["content"][:400] + "...")
                            if idx < len(sources_data) - 1:
                                st.divider()

                    st.caption(f"Задержка: {latency:.2f}с | Модель: {selected_model}")

                except requests.RequestException as e:
                    st.error(f"Ошибка подключения к API: {e}")
                    st.info("Убедитесь, что FastAPI запущен: uvicorn api:app --reload")


if __name__ == "__main__":
    main()
