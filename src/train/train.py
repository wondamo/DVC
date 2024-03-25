from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential


def compile_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=300, trainable=True, name="Input"))
    model.add(Dense(300, name="Dense1"))
    model.add(Dropout(rate=0.25, name="Dropout1"))
    model.add(Dense(128, name="Dense2"))
    model.add(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15, name="LSTM"))
    model.add(GRU(64, return_sequences=False, dropout=0.15, name="GRU"))
    model.add(Dense(64, name="Dense3"))
    model.add(Dropout(rate=0.15, name="Dropout2"))
    model.add(Dense(32, name="Dense4"))
    model.add(Dense(1, activation="sigmoid", name="Output"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def train_model(X_train, y_train, vocab_size, config):
    model = compile_model(vocab_size)
    model.summary()

    # Implement callbacks to handle overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_save = ModelCheckpoint(config['train']['model_checkpoint'], save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=config['train']['model_parameters']['batch_size'], 
        epochs=config['train']['model_parameters']['epochs'], 
        validation_split=config['train']['model_parameters']['validation_split'], 
        callbacks=[early_stopping, model_save]
    )

    return (model, history)