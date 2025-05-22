import os
from model.lstm_model import LSTMModel
from model.cnn_lstm_model import CNNLSTMModel
from model.bidirectional_lstm_model import BidirectionalLSTMModel
from model.gru_model import GRUModel
from model.transformer_model import TransformerModel
from tensorflow.keras.models import load_model
import numpy as np

def generate_music(model_type, duration=500, temperature=0.7):
    # Маппинг имен моделей к классам
    model_map = {
        'lstm': LSTMModel,
        'cnn_lstm': CNNLSTMModel,
        'bidirectional_lstm': BidirectionalLSTMModel,
        'gru': GRUModel,
        'transformer': TransformerModel
    }
    
    # Маппинг имен моделей к именам файлов
    model_file_map = {
        'lstm': 'lstm_model.h5',
        'cnn_lstm': 'cnn_lstm_model.h5',
        'bidirectional_lstm': 'bidirectional lstm_model.h5',
        'gru': 'gru_model.h5',
        'transformer': 'transformer_model.h5'
    }
    
    # Создаем директорию для выходных файлов, если её нет
    os.makedirs("output", exist_ok=True)
    
    # Инициализируем генератор
    generator = model_map[model_type]()
    
    # Загружаем веса модели
    weights_path = os.path.join('src', 'model', 'model_weights', model_file_map[model_type])
    if os.path.exists(weights_path):
        print(f'Загрузка весов из {weights_path}')
        if generator.load_model(weights_path):
            print('Модель успешно загружена!')
        else:
            print('Не удалось загрузить модель')
            return
    else:
        print(f'Модель не найдена по пути: {weights_path}')
        return
    
    # Генерируем музыку
    output_path = os.path.join('output', f'generated_{model_type}.mid')
    print(f'Генерация музыки...')
    try:
        generator.generate_music(
            output_path,
            length=duration,
            temperature=temperature
        )
        print(f'Музыка успешно сгенерирована и сохранена в {output_path}')
    except Exception as e:
        print(f'Ошибка при генерации музыки: {str(e)}')

if __name__ == "__main__":
    # Пример использования
    model_type = 'cnn_lstm'  # Можно выбрать: 'lstm', 'cnn_lstm', 'bidirectional_lstm', 'gru', 'transformer'
    duration = 100  # Длительность в нотах
    temperature = 0.7  # Температура генерации (0.1 - 1.0)
    
    generate_music(model_type, duration, temperature) 