import streamlit as st
import pandas as pd
import io
from ml.model import TenderRiskModel, FEATURE_COLS

st.set_page_config(page_title="Tender Risk AI Pro", page_icon="🔍", layout="wide")

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
    # Здесь остается логика из предыдущего сообщения с ползунками
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        st.info("Используйте боковую панель или введите данные здесь для быстрой проверки.")
        