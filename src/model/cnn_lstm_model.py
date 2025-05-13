from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from .base_model import BaseMusicModel

class CNNLSTMModel(BaseMusicModel):
    def _build_model(self):
        """Создание CNN-LSTM модели"""
        if self.n_vocab is None:
            raise ValueError("Размер словаря не определен. Сначала вызовите prepare_sequences")
            
        model = Sequential([
            # Сверточные слои для извлечения признаков
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, 1)),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            # LSTM слои для обработки последовательностей
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
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