
#не запускать app.py
#Иван Сергеевич тести все в generate_music.py
#к пятнице результат генерации  Будет лучше у меня ноут не вывозит столько обучать просто тут у меня с железом в целом пробемы(((

import streamlit as st
import os
from model.lstm_model import LSTMModel
from model.cnn_lstm_model import CNNLSTMModel
from model.bidirectional_lstm_model import BidirectionalLSTMModel
from model.gru_model import GRUModel
from model.transformer_model import TransformerModel
import pretty_midi
import numpy as np
import tempfile
from tensorflow.keras.models import load_model

# Настройка страницы
st.set_page_config(
    page_title="Генератор музыки",
    page_icon="🎵",
    layout="wide"
)

# Заголовок
st.title("🎵 Генератор музыки")

# Создаем директории, если они не существуют
os.makedirs("output", exist_ok=True)
os.makedirs(os.path.join('src', 'model', 'model_weights'), exist_ok=True)

# Выбор модели
model_type = st.sidebar.selectbox(
    'Выберите модель',
    ['LSTM', 'CNN-LSTM', 'Bidirectional LSTM', 'GRU', 'Transformer']
)

# Инициализация генератора в зависимости от выбранной модели
model_map = {
    'LSTM': LSTMModel,
    'CNN-LSTM': CNNLSTMModel,
    'Bidirectional LSTM': BidirectionalLSTMModel,
    'GRU': GRUModel,
    'Transformer': TransformerModel
}

# Маппинг имен моделей к именам файлов
model_file_map = {
    'LSTM': 'lstm_model.h5',
    'CNN-LSTM': 'cnn_lstm_model.h5',
    'Bidirectional LSTM': 'bidirectional lstm_model.h5',
    'GRU': 'gru_model.h5',
    'Transformer': 'transformer_model.h5'
}

generator = model_map[model_type]()

# Загрузка весов модели
weights_path = os.path.join('src', 'model', 'model_weights', model_file_map[model_type])
if os.path.exists(weights_path):
    st.info(f'Попытка загрузки весов из {weights_path}')
    try:
        # Пробуем загрузить веса напрямую
        model = load_model(weights_path)
        # Копируем веса в генератор
        generator.model.set_weights(model.get_weights())
        st.success('Модель успешно загружена!')
    except Exception as e:
        st.warning(f'Не удалось загрузить веса модели. Ошибка: {str(e)}')
        st.info('Будет использована новая модель.')
else:
    st.warning(f'Модель не найдена по пути: {weights_path}. Пожалуйста, сначала обучите модель.')

# Параметры генерации
st.subheader('Параметры генерации')
duration = st.slider('Длительность (ноты)', min_value=100, max_value=1000, value=500)
temperature = st.slider('Температура', min_value=0.1, max_value=1.0, value=0.7, step=0.1)

if st.button('Сгенерировать музыку'):
    with st.spinner('Генерация музыки...'):
        try:
            # Создаем временный файл для MIDI
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                output_path = tmp.name
            
            # Генерируем музыку
            generator.generate_music(
                output_path,
                length=duration,
                temperature=temperature
            )
            
            # Отображаем результат
            st.success('Музыка сгенерирована!')
            
            # Создаем кнопку для скачивания
            with open(output_path, 'rb') as f:
                st.download_button(
                    label='Скачать MIDI файл',
                    data=f,
                    file_name='generated_music.mid',
                    mime='audio/midi'
                )
            
            # Удаляем временный файл
            os.unlink(output_path)
        except Exception as e:
            st.error(f'Ошибка при генерации музыки: {str(e)}')

# Информация о проекте
st.header("О проекте")
st.markdown(f"""
Этот проект представляет собой генератор музыки, использующий различные архитектуры нейронных сетей для создания MIDI-файлов.

### Текущая модель: {model_type}
- Используется для генерации последовательностей нот
- Оптимизирована для работы с музыкальными данными

### Технологии:
- TensorFlow/Keras для нейронных сетей
- PrettyMIDI для работы с MIDI файлами
- Streamlit для веб-интерфейса

### Доступные архитектуры:
1. LSTM - классическая архитектура для работы с последовательностями
2. CNN-LSTM - комбинация сверточных и рекуррентных слоев
3. Bidirectional LSTM - двунаправленная обработка последовательностей
4. GRU - упрощенная версия LSTM
5. Transformer - современная архитектура с механизмом внимания

### Как это работает:
1. Модель обучается на последовательностях нот из MIDI файлов
2. При генерации модель создает новые последовательности нот
3. Результат сохраняется в MIDI файл, который можно прослушать или скачать
""")

try:
    model = load_model('src/model/model_weights/best_model.h5')
    print("Файл весов успешно загружен.")
except Exception as e:
    print(f"Ошибка при загрузке файла весов: {str(e)}") 

