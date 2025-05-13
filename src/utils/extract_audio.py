import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

def extract_audio_from_tfrecord(tfrecord_path, output_dir, max_files=1000):
    """
    Извлекает аудио файлы из tfrecord файла NSynth
    """
    # Создаем директорию для аудио файлов
    os.makedirs(output_dir, exist_ok=True)
    
    # Функция для парсинга tfrecord
    def parse_tfrecord(example_proto):
        features = {
            'audio': tf.io.FixedLenFeature([64000], tf.float32),
            'pitch': tf.io.FixedLenFeature([], tf.int64),
            'velocity': tf.io.FixedLenFeature([], tf.int64),
            'instrument_family': tf.io.FixedLenFeature([], tf.int64),
            'instrument_source': tf.io.FixedLenFeature([], tf.int64),
            'note': tf.io.FixedLenFeature([], tf.int64),
            'qualities': tf.io.FixedLenFeature([10], tf.int64),
            'sample_rate': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, features)
    
    # Создаем датасет из tfrecord файла
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    # Извлекаем аудио файлы
    count = 0
    for example in tqdm(dataset, desc="Извлечение аудио файлов"):
        if count >= max_files:
            break
            
        # Получаем данные
        audio = example['audio'].numpy()
        pitch = example['pitch'].numpy()
        instrument_family = example['instrument_family'].numpy()
        
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        
        # Создаем имя файла
        filename = f'instrument_{instrument_family}_pitch_{pitch}_{count}.wav'
        output_path = os.path.join(output_dir, filename)
        
        # Сохраняем аудио файл
        sf.write(output_path, audio, 16000)
        
        count += 1
        
    print(f'Извлечено {count} аудио файлов в {output_dir}')

if __name__ == '__main__':
    # Пути к файлам
    tfrecord_path = 'data/nsynth-test.tfrecord'
    output_dir = 'data/audio'
    
    # Извлекаем аудио файлы
    extract_audio_from_tfrecord(tfrecord_path, output_dir) 