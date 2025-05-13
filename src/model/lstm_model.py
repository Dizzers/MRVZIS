from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from .base_model import BaseMusicModel

class LSTMModel(BaseMusicModel):
    def _build_model(self):
        """Создание LSTM модели"""
        if self.n_vocab is None:
            raise ValueError("Размер словаря не определен. Сначала вызовите prepare_sequences")
            
        model = Sequential([
            # Первый LSTM слой
            LSTM(128, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # Второй LSTM слой
            LSTM(64),
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