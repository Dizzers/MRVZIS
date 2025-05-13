from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from .base_model import BaseMusicModel

class GRUModel(BaseMusicModel):
    def _build_model(self):
        """Создание GRU модели"""
        if self.n_vocab is None:
            raise ValueError("Размер словаря не определен. Сначала вызовите prepare_sequences")
            
        model = Sequential([
            # Первый GRU слой
            GRU(256, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Второй GRU слой
            GRU(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Третий GRU слой
            GRU(64),
            Dropout(0.2),
            BatchNormalization(),
            
            # Полносвязные слои
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.n_vocab, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model 