from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from .base_model import BaseMusicModel

class BidirectionalLSTMModel(BaseMusicModel):
    def _build_model(self):
        """Создание Bidirectional LSTM модели"""
        if self.n_vocab is None:
            raise ValueError("Размер словаря не определен. Сначала вызовите prepare_sequences")
            
        model = Sequential([
            # Первый двунаправленный LSTM слой
            Bidirectional(LSTM(128, return_sequences=True), 
                         input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Второй двунаправленный LSTM слой
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            BatchNormalization(),
            
            # Третий двунаправленный LSTM слой
            Bidirectional(LSTM(32)),
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