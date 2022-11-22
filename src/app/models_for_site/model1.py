from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow import keras
from tensorflow import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
try:
    from .scaler import Scaler
except:
    from scaler import Scaler

def get_model_for_regression(input_shape = 28*28):
    model = Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(.1))
    model.add(Dense(12, activation="relu"))
    model.add(Dropout(.1))
    model.add(Dense(8, activation="relu"))

    model.add(Dense(1, activation="relu"))
    model.build()
    model.summary()

    model.compile(loss="mean_squared_error",
                  optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model

def get_model_for_regression_FunkAPI(input_shape = 28*28):
    input = keras.Input(shape=(input_shape,))

    #x_out = Dense(16, activation="relu")(input)
    #x_out = math.sin(Dense(16, activation="relu")(x_out))
    #x_out = Dense(16, activation="relu")(x_out)

    x = Dense(8, activation="relu")(input)
    x1 = Dense(8, activation="relu")(x)
    x2 = Dense(8, activation="tanh")(x)
    b = Dense(8, activation="relu")(x)
    x = Dense(8, activation="relu")(x1 * x2 + b)


    output = Dense(1, activation="relu")(x)  #x_out +
    model = keras.Model(inputs=input, outputs=output, name="funkAPI_model")

    model.summary()

    model.compile(loss="mean_squared_error",
                  optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model


def get_data_for_regression(df: pd.DataFrame , name_y: str ):

    df = df.applymap(lambda x: x.replace(",", ".") if isinstance(x, str) else x)
    df.head()
    possible_y = ["Соотношение матрица-наполнитель", "Модуль упругости при растяжении, ГПа", "Прочность при растяжении, МПа"]
    y = df[name_y].astype("float64")
    df = df.drop(columns=possible_y )

    encoder = LabelEncoder()
    df["Угол нашивки, град"] = encoder.fit_transform(df["Угол нашивки, град"].values)
    scaler = Scaler()
    X = scaler.fit_transform(df.values.astype("float64"), True)
    return train_test_split(X, y)

def main_processing(df: pd.DataFrame = None, name_y: str= None):
    if name_y is None:
        name_y = "Прочность при растяжении, МПа"  #"Модуль упругости при растяжении, ГПа"
    if df is None:
        df = pd.read_csv(r"data/dataForFinal.csv")

    x_train, x_test, y_train,  y_test = get_data_for_regression(df, name_y)
    model = get_model_for_regression(input_shape = x_train.shape[1])  #_FunkAPI
    batch_size = 3
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        encoding='utf-8',
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch')
    es_callback = keras.callbacks.EarlyStopping(
        # Прекратить обучение если `val_loss` больше не улучшается
        monitor='val_loss',
        # "больше не улучшается" определим как "не лучше чем 1e-2 и меньше"
        min_delta=1e-2,
        # "больше не улучшается" далее определим как "как минимум в течение 2 эпох"
        patience=10,
        verbose=1)
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1400,
                        callbacks=[cp_callback, es_callback],  # , es_callback
                        verbose=1,
                        validation_data=(x_test, y_test))
    model.save('my_model.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    pr = model.predict(x_test).ravel()
    rel = np.abs(pr - y_test)/y_test
    print(f"Test relative error, mean: {np.mean(rel)}, max: {np.max(rel)}")
    return {"rel": rel.tolist(), "pr": pr.tolist(), "y_test": y_test.tolist(), "model_path": "static/my_model.h5", "N": pr.shape[0]}

if __name__ == "__main__":
    main_processing()




