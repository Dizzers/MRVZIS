import pretty_midi
import soundfile as sf
import numpy as np
import os

def midi_to_audio(midi_path, output_path, sample_rate=44100):
    """
    Конвертирует MIDI файл в аудио файл (WAV)
    
    Args:
        midi_path (str): Путь к MIDI файлу
        output_path (str): Путь для сохранения аудио файла
        sample_rate (int): Частота дискретизации
    """
    # Загрузка MIDI файла
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Синтез аудио
    audio_data = pm.synthesize(fs=sample_rate)
    
    # Нормализация аудио
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Сохранение в WAV файл
    sf.write(output_path, audio_data, sample_rate)

def batch_convert_midi_to_audio(input_dir, output_dir, sample_rate=44100):
    """
    Конвертирует все MIDI файлы в директории в аудио файлы
    
    Args:
        input_dir (str): Директория с MIDI файлами
        output_dir (str): Директория для сохранения аудио файлов
        sample_rate (int): Частота дискретизации
    """
    # Создание выходной директории, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Конвертация всех MIDI файлов
    for filename in os.listdir(input_dir):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            midi_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.mid', '.wav').replace('.midi', '.wav'))
            
            try:
                midi_to_audio(midi_path, output_path, sample_rate)
                print(f"Успешно конвертирован {filename}")
            except Exception as e:
                print(f"Ошибка при конвертации {filename}: {str(e)}") 