import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
import gc
from music21 import converter, instrument, note, chord, stream
import pretty_midi

class BaseMusicModel:
    def __init__(self, sequence_length=32):
        self.sequence_length = sequence_length
        self.notes = None
        self.pitchnames = None
        self.n_vocab = None
        self.model = None
        
    def _build_model(self):
        """Метод должен быть переопределен в дочерних классах"""
        raise NotImplementedError("Метод _build_model должен быть реализован в дочернем классе")
    
    def prepare_sequences(self, notes):
        """Подготовка последовательностей для обучения"""
        self.pitchnames = sorted(set(item for item in notes))
        self.n_vocab = len(self.pitchnames)
        print(f"Количество уникальных нот: {self.n_vocab}")
        
        self.model = self._build_model()
        
        note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        
        network_input = []
        network_output = []
        
        step = 4
        
        notes_array = np.array([note_to_int[note] for note in notes])
        
        for i in range(0, len(notes) - self.sequence_length, step):
            sequence_in = notes_array[i:i + self.sequence_length]
            sequence_out = notes_array[i + self.sequence_length]
            network_input.append(sequence_in)
            network_output.append(sequence_out)
        
        network_input = np.array(network_input)
        network_input = np.reshape(network_input, (len(network_input), self.sequence_length, 1))
        network_output = to_categorical(network_output, num_classes=self.n_vocab)
        
        return network_input, network_output
    
    def save_model(self, weights_path):
        """Сохранение модели и словаря нот"""
        try:
            print("\nНачинаем сохранение модели...")
            
            if self.model is None:
                print("ОШИБКА: модель не создана")
                return False
                
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            print("1. Сохраняем веса модели...")
            self.model.save_weights(weights_path)
            print("   Веса модели сохранены успешно")
            
            print("2. Сохраняем словарь нот...")
            pitchnames_path = weights_path.replace('.h5', '_pitchnames.npy')
            np.save(pitchnames_path, self.pitchnames)
            print("   Словарь нот сохранен успешно")
            
            print("Модель успешно сохранена!\n")
            return True
        except Exception as e:
            print(f"\nОШИБКА при сохранении модели: {str(e)}")
            return False
    
    def load_model(self, weights_path):
        """Загрузка модели и словаря нот"""
        try:
            print("\nНачинаем загрузку модели...")
            
            if not os.path.exists(weights_path):
                print(f"ОШИБКА: Файл модели не найден: {weights_path}")
                return False
                
            pitchnames_path = weights_path.replace('.h5', '_pitchnames.npy')
            if not os.path.exists(pitchnames_path):
                print(f"ОШИБКА: Файл словаря нот не найден: {pitchnames_path}")
                return False
            
            print("1. Загружаем словарь нот...")
            self.pitchnames = np.load(pitchnames_path, allow_pickle=True)
            self.n_vocab = len(self.pitchnames)
            print(f"   Словарь нот загружен. Размер словаря: {self.n_vocab}")
            
            print("2. Создаем новую модель...")
            self.model = self._build_model()
            print("   Модель создана успешно")
            
            print("3. Загружаем веса модели...")
            self.model.load_weights(weights_path)
            print("   Веса модели загружены успешно")
            
            print("Модель успешно загружена!\n")
            return True
            
        except Exception as e:
            print(f"\nОШИБКА при загрузке модели: {str(e)}")
            self.model = None
            self.pitchnames = None
            self.n_vocab = None
            return False

    def generate_music(self, output_path, length=500, temperature=0.7):
        """Генерация музыки"""
        if self.model is None or self.pitchnames is None:
            raise ValueError("Модель не загружена. Сначала загрузите модель.")

        # Создаем словарь для преобразования индексов в ноты
        int_to_note = dict((number, note) for number, note in enumerate(self.pitchnames))
        
        # Начинаем с случайной последовательности
        start = np.random.randint(0, len(self.pitchnames) - self.sequence_length)
        pattern = [int_to_note[value] for value in np.random.randint(0, self.n_vocab, self.sequence_length)]
        
        # Генерируем ноты
        prediction_output = []
        
        for _ in range(length):
            # Подготавливаем входные данные
            prediction_input = np.zeros((1, self.sequence_length, 1))
            for i, note in enumerate(pattern):
                prediction_input[0, i, 0] = np.where(self.pitchnames == note)[0][0]
            
            # Получаем предсказание
            prediction = self.model.predict(prediction_input, verbose=0)[0]
            
            # Применяем температуру
            prediction = np.log(prediction) / temperature
            exp_preds = np.exp(prediction)
            pred_probs = exp_preds / np.sum(exp_preds)
            
            # Выбираем следующую ноту
            index = np.random.choice(len(pred_probs), p=pred_probs)
            result = int_to_note[index]
            prediction_output.append(result)
            
            # Обновляем паттерн
            pattern.append(result)
            pattern = pattern[1:]
        
        # Создаем MIDI файл
        pm = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.Instrument(program=0)  # 0 = акустическое пиано
        
        current_time = 0.0
        for note_name in prediction_output:
            try:
                # Парсим ноту и длительность
                if '_' in note_name:
                    note_parts = note_name.split('_')
                    note_value = note_parts[0]
                    duration_str = note_parts[1]
                    
                    # Преобразуем длительность
                    if '/' in duration_str:
                        num, denom = map(int, duration_str.split('/'))
                        duration = float(num) / float(denom)
                    else:
                        duration = float(duration_str)
                    
                    # Определяем высоту ноты
                    try:
                        # Пробуем преобразовать как числовое значение
                        if '.' in note_value:
                            # Если это несколько нот через точку, берем первую
                            pitch = int(float(note_value.split('.')[0]))
                        else:
                            pitch = int(float(note_value))
                    except ValueError:
                        # Если не получилось, пробуем как музыкальную ноту
                        from music21 import note as m21_note
                        note_obj = m21_note.Note(note_value)
                        pitch = note_obj.pitch.midi
                else:
                    # Если нота без длительности, используем формат music21
                    from music21 import note as m21_note
                    note_obj = m21_note.Note(note_name)
                    pitch = note_obj.pitch.midi
                    duration = 0.5  # По умолчанию половинная нота
                
                # Проверяем, что pitch находится в допустимом диапазоне MIDI (0-127)
                pitch = max(0, min(127, pitch))
                
                # Создаем ноту
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )
                piano_program.notes.append(note)
                current_time += duration
            except Exception as e:
                print(f"Пропуск ноты {note_name} из-за ошибки: {str(e)}")
                continue
        
        pm.instruments.append(piano_program)
        pm.write(output_path) 