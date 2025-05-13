import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from music21 import converter, instrument, note, chord
import gc
from .lstm_model import LSTMModel
from .cnn_lstm_model import CNNLSTMModel
from .bidirectional_lstm_model import BidirectionalLSTMModel
from .gru_model import GRUModel
from .transformer_model import TransformerModel

def train_and_evaluate_models(midi_files, epochs=20, batch_size=64, max_files=15, validation_split=0.15):
    """Обучение и оценка всех моделей"""
    results = {}
    
    # Список всех моделей для сравнения (Transformer первым)
    models = {
        'Transformer': TransformerModel(),
        'LSTM': LSTMModel(),
        'CNN-LSTM': CNNLSTMModel(),
        'Bidirectional LSTM': BidirectionalLSTMModel(),
        'GRU': GRUModel()
    }
    
    # Подготовка данных
    print('Подготовка данных для обучения...')
    all_notes = []
    
    # Используем многопроцессорную обработку для ускорения загрузки данных
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def process_midi_file(file):
        try:
            midi = converter.parse(file)
            notes_to_parse = None
            
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes
            
            file_notes = []
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    duration = element.quarterLength
                    file_notes.append(f"{str(element.pitch)}_{duration}")
                elif isinstance(element, chord.Chord):
                    duration = element.quarterLength
                    chord_notes = '.'.join(str(n) for n in element.normalOrder)
                    file_notes.append(f"{chord_notes}_{duration}")
            
            return file_notes
        except Exception as e:
            print(f'Ошибка при обработке файла {file}: {str(e)}')
            return []
    
    # Параллельная обработка файлов с увеличенным количеством воркеров
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_midi_file, file) for file in midi_files[:max_files]]
        for future in futures:
            all_notes.extend(future.result())
    
    if not all_notes:
        raise ValueError("Не удалось извлечь ноты из файлов")
    
    print(f'Всего извлечено нот: {len(all_notes)}')
    
    # Обучение и оценка каждой модели
    for model_name, model in models.items():
        print(f'\nОбучение модели {model_name}...')
        
        # Подготовка последовательностей
        network_input, network_output = model.prepare_sequences(all_notes)
        
        # Оптимизированные callbacks
        weights_path = f'src/model/model_weights/{model_name.lower().replace("-", "_")}_model.h5'
        callbacks = [
            ModelCheckpoint(
                weights_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,  # Уменьшаем patience для более быстрой остановки
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,  # Уменьшаем patience для более быстрой адаптации learning rate
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Обучение модели с оптимизированными параметрами
        history = model.model.fit(
            network_input,
            network_output,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,  # Включаем перемешивание данных
            use_multiprocessing=True,  # Используем многопроцессорную обработку
            workers=4,  # Количество рабочих процессов
            max_queue_size=10  # Ограничиваем размер очереди для предотвращения переполнения памяти
        )
        
        # Сохранение результатов
        results[model_name] = {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'history': history.history
        }
        
        # Сохранение модели
        model.save_model(weights_path)
        
        # Очистка памяти
        del model
        gc.collect()
    
    return results

def print_comparison_results(results):
    """Вывод результатов сравнения моделей"""
    print("\nРезультаты сравнения моделей:")
    print("-" * 80)
    print(f"{'Модель':<20} {'Loss':<10} {'Val Loss':<10} {'Accuracy':<10} {'Val Accuracy':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['final_loss']:<10.4f} "
              f"{metrics['final_val_loss']:<10.4f} "
              f"{metrics['final_accuracy']:<10.4f} "
              f"{metrics['final_val_accuracy']:<10.4f}")
    
    print("-" * 80)
    
    # Находим лучшую модель по валидационной точности
    best_model = max(results.items(), key=lambda x: x[1]['final_val_accuracy'])
    print(f"\nЛучшая модель: {best_model[0]}")
    print(f"Валидационная точность: {best_model[1]['final_val_accuracy']:.4f}")
    print(f"Валидационная loss: {best_model[1]['final_val_loss']:.4f}") 