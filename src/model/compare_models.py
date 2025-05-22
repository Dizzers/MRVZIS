import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from music21 import converter, instrument, note, chord
import gc
import matplotlib.pyplot as plt
from .lstm_model import LSTMModel
from .cnn_lstm_model import CNNLSTMModel
from .bidirectional_lstm_model import BidirectionalLSTMModel
from .gru_model import GRUModel
from .transformer_model import TransformerModel

class LocalMinimumEscape(Callback):
    """Callback для выхода из локального минимума"""
    def __init__(self, patience=10, factor=2.0, min_lr=0.00001, max_lr=0.1):
        super(LocalMinimumEscape, self).__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = float('inf')
        self.escape_count = 0
        self.max_escapes = 3

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.escape_count < self.max_escapes:
                    old_lr = float(self.model.optimizer.lr.numpy())
                    new_lr = min(old_lr * self.factor, self.max_lr)
                    self.model.optimizer.lr.assign(new_lr)
                    print(f'\nПопытка выхода из локального минимума {self.escape_count + 1}/{self.max_escapes}')
                    print(f'Увеличиваем learning rate с {old_lr} до {new_lr}')
                    self.wait = 0
                    self.escape_count += 1
                else:
                    print('\nДостигнуто максимальное количество попыток выхода из локального минимума')

def train_and_evaluate_models(midi_files, epochs=50, batch_size=64, max_files=15, validation_split=0.15):
    """Обучение и оценка всех моделей"""
    print(f'Получено количество эпох в train_and_evaluate_models: {epochs}')
    results = {}
    
    
    models = {
        'LSTM': LSTMModel(),
        'CNN-LSTM': CNNLSTMModel(),
        'Bidirectional LSTM': BidirectionalLSTMModel(),
        'GRU': GRUModel(),
        'Transformer': TransformerModel(),
    }
    
    # Подготовка данных
    print('Подготовка данных для обучения...')
    all_notes = []
    
    
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
    
    # Создаем директорию для сохранения графиков
    os.makedirs('src/model/training_plots', exist_ok=True)
    
    # Обучение и оценка каждой модели
    for model_name, model in models.items():
        print(f'\nОбучение модели {model_name}...')
        print(f'Количество эпох: {epochs}')  # Добавляем отладочный вывод
        
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
            )
        ]
        
        # Обучение модели с оптимизированными параметрами
        print(f'Количество эпох перед model.fit: {epochs}')
        history = model.model.fit(
            network_input,
            network_output,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            use_multiprocessing=True,
            workers=4,
            max_queue_size=10
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
        
        # Построение и сохранение графиков
        plot_training_history(history, model_name)
        
        # Очистка памяти
        del model
        gc.collect()
    
    return results

def plot_training_history(history, model_name):
    """Построение графиков процесса обучения"""
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Точность на обучающей выборке')
    ax1.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
    ax1.set_title(f'Точность модели {model_name}')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # График ошибки
    ax2.plot(history.history['loss'], label='Ошибка на обучающей выборке')
    ax2.plot(history.history['val_loss'], label='Ошибка на валидационной выборке')
    ax2.set_title(f'Ошибка модели {model_name}')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Ошибка')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Настройка отступов
    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig(f'src/model/training_plots/{model_name.lower().replace("-", "_")}_training_history.png')
    plt.close()

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
    
    # Построение сравнительного графика
    plot_comparison_chart(results)

def plot_comparison_chart(results):
    """Построение сравнительного графика всех моделей"""
    models = list(results.keys())
    val_accuracies = [results[model]['final_val_accuracy'] for model in models]
    val_losses = [results[model]['final_val_loss'] for model in models]
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График точности
    ax1.bar(models, val_accuracies)
    ax1.set_title('Сравнение валидационной точности моделей')
    ax1.set_xlabel('Модель')
    ax1.set_ylabel('Валидационная точность')
    ax1.grid(True, axis='y')
    
    # График ошибки
    ax2.bar(models, val_losses)
    ax2.set_title('Сравнение валидационной ошибки моделей')
    ax2.set_xlabel('Модель')
    ax2.set_ylabel('Валидационная ошибка')
    ax2.grid(True, axis='y')
    
    # Настройка отступов
    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig('src/model/training_plots/models_comparison.png')
    plt.close() 