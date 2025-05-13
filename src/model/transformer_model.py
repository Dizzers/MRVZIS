import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .base_model import BaseMusicModel

class TransformerModel(BaseMusicModel):
    def __init__(self):
        super().__init__()
        self.sequence_length = 32
        self.embedding_dim = 256
        self.num_heads = 8
        self.ff_dim = 512
        self.num_transformer_blocks = 4
        self.dropout_rate = 0.1
        self.model = None
        self.note_to_int = None
        self.int_to_note = None

    def _build_model(self, input_shape, output_shape):
        # Входной слой
        inputs = layers.Input(shape=(self.sequence_length,))
        
        # Слой эмбеддинга
        x = layers.Embedding(output_shape, self.embedding_dim)(inputs)
        
        # Добавляем позиционное кодирование
        x = self._add_positional_encoding(x)
        
        # Transformer блоки
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_block(x)
        
        # Глобальный пулинг
        x = layers.GlobalAveragePooling1D()(x)
        
        # Полносвязные слои
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Выходной слой
        outputs = layers.Dense(output_shape, activation='softmax')(x)
        
        # Создаем модель
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Компилируем модель
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model

    def _add_positional_encoding(self, x):
        # Создаем позиционное кодирование
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * -(np.log(10000.0) / self.embedding_dim))
        
        # Создаем синусоидальное кодирование
        pos_encoding = np.zeros((self.sequence_length, self.embedding_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Преобразуем в тензор и добавляем размерность батча
        pos_encoding = tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        # Добавляем позиционное кодирование к входным данным
        return x + pos_encoding

    def _transformer_block(self, x):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(self.ff_dim, activation='relu')(x)
        ffn_output = layers.Dense(self.embedding_dim)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x

    def prepare_sequences(self, notes):
        """Подготовка последовательностей для обучения"""
        # Получаем уникальные ноты
        unique_notes = sorted(set(notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
        self.int_to_note = dict((number, note) for number, note in enumerate(unique_notes))
        
        # Создаем входные последовательности и выходные значения
        network_input = []
        network_output = []
        
        # Создаем последовательности
        for i in range(0, len(notes) - self.sequence_length, 1):
            sequence_in = notes[i:i + self.sequence_length]
            sequence_out = notes[i + self.sequence_length]
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
        
        # Преобразуем в numpy массивы
        network_input = np.array(network_input)
        network_output = tf.keras.utils.to_categorical(network_output, num_classes=len(unique_notes))
        
        # Инициализируем модель
        self.model = self._build_model(
            input_shape=(self.sequence_length,),
            output_shape=len(unique_notes)
        )
        
        return network_input, network_output 