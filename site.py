import streamlit as st
import pickle
import sklearn
import pandas as pd
sklearn.set_config(transform_output="pandas")


# Загрузка модели и предобработчика
@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('pipeline.pkl', 'rb') as f:
        prepocessor = pickle.load(f)
    return model, prepocessor

model, prepocessor = load_model()
def preprocess_input(data):
    # Применяем предобработку к входным данным
    preprocessed_data = prepocessor.transform(data)
    return preprocessed_data

def predict_heart_disease(data):
    # Предобработка входных данных
    preprocessed_data = preprocess_input(data)
    # Предсказание с использованием модели
    prediction = model.predict(preprocessed_data)
    return prediction

# Создаем интерфейс Streamlit
st.title('Прогнозирование сердечных заболеваний')

# Создаем форму для ввода данных
st.sidebar.header('Введите данные пациента:')
age = st.sidebar.slider('Возраст', 20, 80, 40)
sexdict={'М' :"M", 'Ж' :"F"}
sex = sexdict[ st.sidebar.radio('Пол', ['М', 'Ж'])]

chest_pain_types = {
    'TA': 'Типичная стенокардия (Typical Angina): боль, связанная с физической активностью или стрессом и облегчающаяся в покое.',
    'ATA': 'Атипичная стенокардия (Atypical Angina): боль в груди, не соответствующая характеру типичной стенокардии.',
    'NAP': 'Неангинальная боль в груди (Non-Anginal Pain): боли, не вызванные ишемией (недостаточным кровоснабжением сердца).',
    'ASY': 'Асимптоматическая (Asymptomatic): отсутствие симптомов боли в груди.',
}
chest_pain_type = st.sidebar.selectbox('Тип боли в груди', list(chest_pain_types.keys()), help="Выберите тип боли в груди")
chest_pain_type_description = chest_pain_types.get(chest_pain_type, 'Нет описания')
# Отображаем пояснение для выбранного типа боли
st.sidebar.markdown(f'Описание: {chest_pain_type_description}')

resting_bp = st.sidebar.slider('Давление в покое', 80, 200, 120)
cholesterol = st.sidebar.slider('Холестерин', 100, 400, 200)


fasting_bs_description = """
**Уровень сахара натощак (Fasting Blood Sugar):**
- **1:** Уровень сахара в крови натощак более 120 мг/дл.
- **0:** Уровень сахара в крови натощак не превышает 120 мг/дл.
"""

# Отображаем многострочное описание для выбранного уровня сахара натощак
st.sidebar.markdown(f'\n{fasting_bs_description}')
fasting_bs = st.sidebar.radio('Уровень сахара натощак', ['1', '0'])

resting_ecg_types = {
    'Normal': 'Электрокардиограмма в покое считается нормальной.',
    'ST': 'Электрокардиограмма в покое с присутствием аномалий в ST-T волне (инверсии волны T и/или изменение ST-сегмента).',
    'LVH': 'Электрокардиограмма в покое, показывающая вероятное или определенное утолщение левого желудочка по критериям Эстеса.'
}

resting_ecg = st.sidebar.selectbox('Электрокардиограмма в покое', list(resting_ecg_types.keys()), help="Выберите тип электрокардиограммы в покое")
resting_ecg_description = resting_ecg_types.get(resting_ecg, 'Нет описания')

# Отображаем многострочное описание для выбранного типа электрокардиограммы в покое
st.sidebar.markdown(f'**Описание:**\n{resting_ecg_description}')



max_hr = st.sidebar.slider('Максимальная частота сердечных сокращений', 60, 202, 120)

yesnodict={'Да': 'Y', 'Нет': 'N'}

exercise_angina = st.sidebar.radio('Стенокардия при физической нагрузке', ['Да', 'Нет'])
exercise_angina =yesnodict[ exercise_angina]

# Пояснения к "Старое углубление ST"
oldpeak_description = """
**Старое углубление ST (Oldpeak):**
- Измеряется в миллиметрах.
- Отражает степень депрессии сегмента ST после упражнения.
"""
st.sidebar.markdown(f'**Описание:**\n{oldpeak_description}')
oldpeak = st.sidebar.slider('Старое углубление ST', 0.0, 6.0, 2.0, help="Выберите значение старого углубления ST")


# Пояснения к "Наклон пика упражнения ST"
st_slope_description = """
**Наклон пика упражнения ST (ST_Slope):**
- **Up:** Восходящий наклон.
- **Flat:** Плоский наклон.
- **Down:** Нисходящий наклон.
"""
st.sidebar.markdown(f'**Описание:**\n{st_slope_description}')
st_slope = st.sidebar.selectbox('Наклон пика упражнения ST', ['Up', 'Flat', 'Down'], help="Выберите наклон пика упражнения ST")

# Создаем DataFrame с введенными данными
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope],
})
# Отображаем введенные пользователем данные
st.subheader('Введенные данные:')
st.write(input_data)

# Предсказываем наличие сердечного заболевания

prediction = predict_heart_disease(input_data)
if prediction[0] == 1:
    st.error('Возможно, у него есть сердечное заболевание.')
else:
    st.success('Пациент скорее всего в порядке. ')


