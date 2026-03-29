import streamlit as st
import pandas as pd
import io
from ml.model import TenderRiskModel, FEATURE_COLS

st.set_page_config(page_title="Tender Risk AI Pro", page_icon="🔍", layout="wide")

def get_single_ai_comment(result):
    """Генерация развернутого комментария для одиночного тендера"""
    score = result["final_score"]
    # Берем описание самого подозрительного фактора
    top_feat = result["top_features"][0]["description"] if result["top_features"] else "Неизвестно"
    
    if score >= 75:
        return f"🚨 **Критическая аномалия.** Главный триггер: *{top_feat}*. Высокая вероятность 'заточки' или сговора. Требуется полная блокировка и проверка!"
    elif score >= 50:
        return f"⚠️ **Высокий риск.** Параметры выбиваются из нормы. Основная причина: *{top_feat}*. Рекомендуется ручной аудит документации."
    elif score >= 25:
        return f"🔍 **Средний риск.** Есть небольшие отклонения (влияет: *{top_feat}*). Возможно, это специфика отрасли, но стоит присмотреться."
    else:
        return "✅ **Типовой тендер.** Подозрительных паттернов не обнаружено, всё в рамках рынка."

@st.cache_resource
def load_model():
    return TenderRiskModel.load("ml/model.pkl")

try:
    model = load_model()
except:
    st.error("Файл модели ml/model.pkl не найден! Сначала запустите python train.py")
    st.stop()

st.title("🔍 Tender Risk AI: Аналитический центр")

# Создаем вкладки: одна для ручного ввода, другая для файлов
tab1, tab2 = st.tabs(["📄 Загрузка CSV/Excel", "⌨️ Ручной ввод"])

# --- ВКЛАДКА 1: ЗАГРУЗКА ФАЙЛОВ ---
with tab1:
    st.header("Массовая проверка тендеров")
    uploaded_file = st.file_uploader("Выберите файл со списком тендеров", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Читаем файл
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
            
        st.write(f"Загружено строк: {len(df_input)}")
        
        # Проверяем наличие нужных колонок
        missing = [c for c in FEATURE_COLS if c not in df_input.columns]
        
        if missing:
            st.warning(f"В файле не хватает колонок: {', '.join(missing)}. Модель может работать некорректно.")
        
        if st.button("Запустить анализ", type="primary"):
            with st.spinner('ИИ анализирует данные...'):
                # Делаем предсказание для всей таблицы сразу
                results_df = model.predict_batch(df_input)
                
                # Считаем статистику
                st.success("Анализ завершен!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Всего проверено", len(results_df))
                critical_count = len(results_df[results_df['risk_level'] == 'critical'])
                c2.metric("Критический риск 🚩", critical_count)
                c3.metric("Средний Score", f"{results_df['final_score'].mean():.1f}%")

                # Отображаем таблицу результатов
                st.subheader("Результаты по лотам")
                
                # Раскрашиваем таблицу для удобства
                def color_risk(val):
                    color = '#ff4b4b' if val == 'critical' else '#ffa500' if val == 'high' else '#2eb82e' if val == 'low' else '#f1c40f'
                    return f'background-color: {color}; color: white; font-weight: bold'

                display_cols = ['lot_id', 'final_score', 'risk_level'] + [c for c in results_df.columns if c.startswith('top_')]
                st.dataframe(results_df.style.applymap(color_risk, subset=['risk_level']), use_container_width=True)

                # Кнопка скачивания результата
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Скачать отчет в CSV",
                    csv,
                    "tender_risk_report.csv",
                    "text/csv",
                    key='download-csv'
                )


with tab2:
    st.header("Проверка одиночного тендера")
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        st.info("Заполните ключевые параметры тендера для быстрой проверки:")
        
        with st.form("single_tender_form"):
            # Группа основных признаков
            m_deadline = st.number_input("Минуты до дедлайна (норма > 1440)", 0, 10000, 1440)
            ip_coll = st.selectbox("Совпадение IP участников (0 - нет, 1 - да)", [0, 1], index=0)
            w_rate = st.slider("Исторический Win-rate победителя", 0.0, 1.0, 0.15)
            participants = st.number_input("Число участников", 1, 50, 5)
            price_red = st.slider("Снижение цены от рынка (%)", 0.0, 1.0, 0.15)
            
            # Кнопка отправки формы
            submitted = st.form_submit_button("Анализировать риск", type="primary")

    with col_res:
        if submitted:
            # Собираем данные. Все, что не ввели явно, заполняем нулями для безопасности
            input_data = {f: 0.0 for f in FEATURE_COLS}
            input_data.update({
                "minutes_before_deadline": m_deadline,
                "ip_collision": ip_coll,
                "winner_win_rate": w_rate,
                "participants_count": participants,
                "price_reduction_pct": price_red
            })
            
            result = model.predict(input_data)
            score = result["final_score"]
            level = result["risk_level"]
            
            st.subheader("Вердикт системы")
            
            # Красивая карточка уровня риска
            st.markdown(f"""
                <div class="risk-card {level}">
                    <h3 style="margin-top:0;">Уровень: {level.upper()}</h3>
                    <h1 style="margin:0;">Score: {score:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Комментарий ИИ
            st.markdown("### 🤖 Комментарий ИИ:")
            st.write(get_single_ai_comment(result))
            
            st.markdown("---")
            st.write("**Топ-3 фактора риска (SHAP):**")
            for feat in result["top_features"][:3]:
                icon = "🔺" if feat["direction"] == "risk_up" else "🔹"
                st.write(f"{icon} {feat['description']} (Влияние: {feat['shap_weight']:.3f})")
        else:
            st.write("👈 Введите данные слева и нажмите кнопку анализа.")
        