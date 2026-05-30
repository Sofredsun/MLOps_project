import time

import requests
import streamlit as st

st.set_page_config(page_title="Школьный ИИ-ассистент", layout="wide")

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


@st.cache_data(ttl=30)
def fetch_concept_alerts():
    try:
        response = requests.get(f"{API_URL}/monitoring/concept-alerts", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def render_drift_panel():
    # Блок отображения уведомлений о дрейфе с кнопками "Скачать отчет" и "Переобучение"
    alerts = fetch_alerts()
    concept_alerts = fetch_concept_alerts()

    has_drift = any(a.get("drift_detected") for a in alerts)
    has_concept_drift = any(a.get("concept_drift_detected") for a in concept_alerts)

    # Показываем панель если есть хоть один дрейф
    if has_drift or has_concept_drift:
        if has_drift:
            latest_alert = alerts[0]
            score = latest_alert.get("drift_score", "N/A")
            threshold = latest_alert.get("threshold", "N/A")
            recommendation = latest_alert.get("recommendation", "Проверьте данные")
            timestamp = latest_alert.get("timestamp", "N/A")

            st.error("**Обнаружен дрейф запросов!**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Drift Score", f"{score}")
                st.metric("Threshold", f"{threshold}")
            with col2:
                st.metric("Time", timestamp[:16] if timestamp != "N/A" else "N/A")
                st.info(f"{recommendation}")

    if has_concept_drift:
        latest = concept_alerts[0]
        st.warning("**Обнаружен Concept Drift - падение качества генерации!**")
        issues = latest.get("issues", [])
        for issue in issues:
            st.error(f"{issue}")

        metrics = latest.get("metrics", {})
        cols = st.columns(3)
        if "dislike_rate" in metrics:
            cols[0].metric("Дизлайков", f"{metrics['dislike_rate']:.0%}")
        if "avg_faithfulness" in metrics:
            cols[1].metric("Faithfulness", metrics["avg_faithfulness"])
        if "avg_answer_relevancy" in metrics:
            cols[2].metric("Relevancy", metrics["avg_answer_relevancy"])

        st.info(latest.get("recommendation", ""))

    # Кнопки "Скачать отчет" и "Переобучение"
    st.markdown("---")
    col_report, col_retrain, col_refresh = st.columns([2, 2, 1])

    with col_report:
        if st.button("Скачать отчет", use_container_width=True):
            with st.spinner("Генерирую отчет..."):
                try:
                    resp = requests.get(f"{API_URL}/monitoring/report", timeout=30)
                    resp.raise_for_status()
                    st.download_button(
                        label="Сохранить HTML-отчет",
                        data=resp.content,
                        file_name="drift_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                except requests.RequestException as e:
                    st.error(f"Не удалось получить отчет: {e}")

    with col_retrain:
        _render_retrain_button()

    with col_refresh:
        if st.button(
            "Обновить статус", help="Обновить статус", use_container_width=True
        ):
            st.cache_data.clear()
            st.rerun()

    st.divider()


def _render_retrain_button():
    """Кнопка переобучения со статусом"""
    # Проверяем текущий статус
    retrain_status = st.session_state.get("retrain_status", {})
    is_running = retrain_status.get("status") == "running"

    if is_running:
        # Обновляем статус с бэкенда
        try:
            resp = requests.get(f"{API_URL}/retrain/status", timeout=5)
            current = resp.json()
            st.session_state["retrain_status"] = current
            if current["status"] == "running":
                st.warning("Переобучение выполняется...")
                return
        except requests.RequestException:
            pass

    if st.button("Переобучение", use_container_width=True, disabled=is_running):
        with st.spinner("Запускаю переобучение..."):
            try:
                resp = requests.post(f"{API_URL}/retrain", timeout=10)
                resp.raise_for_status()
                data = resp.json()
                st.session_state["retrain_status"] = {"status": "running"}

                if data.get("status") == "already_running":
                    st.warning("Переобучение уже запущено")
                else:
                    st.success("Переобучение запущено!")
                    st.caption(data.get("message", ""))
            except requests.RequestException as e:
                st.error(f"Ошибка: {e}")

    # Показываем результат прошлого запуска
    if retrain_status.get("status") == "done":
        st.success(f"Завершено: {retrain_status.get('finished_at', '')[:16]}")
    elif retrain_status.get("status") == "error":
        st.error(f"Ошибка {retrain_status.get('message', '')}")


def main():
    # Панель дрейфа (с кнопками отчета и переобучения)
    render_drift_panel()

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
            "Какой номер телефона директора?",
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

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Источники"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**{idx + 1}.{source['source']}**")
                        st.caption(source["content"][:300] + "...")

            # Кнопки лайк/дизлайк только для сообщений ассистента
            if message["role"] == "assistant" and "request_id" in message:
                feedback_key = f"feedback_{message['request_id']}"

                # Показываем кнопки только если ещё не оценено
                if feedback_key not in st.session_state:
                    col1, col2, col3 = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"like_{i}"):
                            try:
                                requests.post(
                                    f"{API_URL}/feedback",
                                    json={
                                        "request_id": message["request_id"],
                                        "question": st.session_state.messages[i - 1][
                                            "content"
                                        ],
                                        "answer": message["content"],
                                        "rating": 1,
                                    },
                                    timeout=10,
                                )
                                st.session_state[feedback_key] = "like"
                                st.rerun()
                            except requests.RequestException:
                                st.error("Не удалось отправить оценку")
                    with col2:
                        if st.button("👎", key=f"dislike_{i}"):
                            try:
                                requests.post(
                                    f"{API_URL}/feedback",
                                    json={
                                        "request_id": message["request_id"],
                                        "question": st.session_state.messages[i - 1][
                                            "content"
                                        ],
                                        "answer": message["content"],
                                        "rating": 0,
                                    },
                                    timeout=10,
                                )
                                st.session_state[feedback_key] = "dislike"
                                st.rerun()
                            except requests.RequestException:
                                st.error("Не удалось отправить оценку")
                else:
                    # Показываем результат оценки
                    if st.session_state[feedback_key] == "like":
                        st.caption("Вы оценили ответ положительно")
                    else:
                        st.caption("Вы оценили ответ отрицательно")

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

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": displayed,
                            "sources": sources_data,
                            "request_id": data["request_id"],
                        }
                    )

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
                else:
                    # Перезапускаем скрипт, чтобы цикл отрисовки сверху
                    # подхватил новое сообщение и отобразил кнопки лайк/дизлайк
                    st.rerun()


if __name__ == "__main__":
    main()
