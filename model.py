# -*- coding: utf-8 -*-
"""TensorFlow LSTM/GRU time-series training and prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import compute_metrics, create_sequences


def train_recurrent_model(
    daily: pd.DataFrame,
    *,
    model_type: str = "LSTM",
    look_back: int = 28,
    units: int = 64,
    layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    progress_callback=None,
) -> dict:
    """Train an LSTM/GRU on a chronological split and return model/evaluation state."""
    if model_type not in {"LSTM", "GRU"}:
        raise ValueError("model_type phải là LSTM hoặc GRU.")
    if look_back < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("look_back, epochs và batch_size phải lớn hơn 0.")
    if layers < 1 or units < 1 or not 0 <= dropout < 1:
        raise ValueError("Cấu hình số lớp, units hoặc dropout không hợp lệ.")

    frame = daily[["date", "sales"]].copy().sort_values("date")
    values = pd.to_numeric(frame["sales"], errors="coerce").to_numpy(dtype=float)
    if len(values) < look_back * 2 + 2:
        raise ValueError("Cần ít nhất 2 cửa sổ look-back dữ liệu để huấn luyện.")
    if not np.isfinite(values).all():
        raise ValueError("Chuỗi doanh số có giá trị thiếu hoặc không hữu hạn.")
    if np.ptp(values) == 0:
        raise ValueError("Chuỗi doanh số không biến động; không thể huấn luyện RNN.")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "Không tìm thấy TensorFlow. Cài các gói trong requirements.txt "
            "để huấn luyện LSTM/GRU."
        ) from exc

    split_index = int(len(values) * (1 - test_ratio))
    if split_index <= look_back or len(values) - split_index < 1:
        raise ValueError("Tập huấn luyện/kiểm định quá ngắn với cấu hình hiện tại.")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(values[:split_index].reshape(-1, 1)).flatten()
    full_scaled = scaler.transform(values.reshape(-1, 1)).flatten()
    X_train, y_train = create_sequences(train_scaled, look_back)
    test_window = full_scaled[split_index - look_back :]
    X_test, y_test = create_sequences(test_window, look_back)
    if not len(X_train) or not len(X_test):
        raise ValueError("Không tạo được chuỗi huấn luyện/kiểm định.")

    X_train = X_train.reshape((-1, look_back, 1))
    X_test = X_test.reshape((-1, look_back, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(look_back, 1)))
    recurrent = keras.layers.LSTM if model_type == "LSTM" else keras.layers.GRU
    for layer_index in range(layers):
        model.add(
            recurrent(
                units,
                return_sequences=layer_index < layers - 1,
            )
        )
        if dropout:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    callbacks = [keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
    if progress_callback:
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_callback(epoch + 1, epochs, logs or {})

        callbacks.append(ProgressCallback())

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    scaled_prediction = model.predict(X_test, verbose=0).reshape(-1, 1)
    prediction = scaler.inverse_transform(scaled_prediction).flatten()
    actual = scaler.inverse_transform(y_test).flatten()
    predicted_dates = frame["date"].iloc[split_index : split_index + len(actual)].to_numpy()

    return {
        "model": model,
        "scaler": scaler,
        "full_scaled": full_scaled,
        "look_back": look_back,
        "history": history.history,
        "metrics": compute_metrics(actual, prediction),
        "test_data": {
            "dates": predicted_dates,
            "actual": actual,
            "predicted": prediction,
        },
        "last_date": pd.Timestamp(frame["date"].iloc[-1]),
        "daily": frame.reset_index(drop=True),
        "model_config": {
            "type": model_type,
            "look_back": look_back,
            "units": units,
            "layers": layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }


def forecast_future(model, last_window, scaler, horizon: int) -> np.ndarray:
    """Recursively forecast a positive number of future daily sales values."""
    if horizon < 1:
        raise ValueError("Số ngày dự báo phải lớn hơn 0.")
    window = np.asarray(last_window, dtype=float).reshape(-1)
    if not len(window):
        raise ValueError("Không có cửa sổ dữ liệu gần nhất để dự báo.")
    output = []
    for _ in range(horizon):
        prediction = float(
            np.asarray(model.predict(window.reshape(1, len(window), 1), verbose=0)).reshape(-1)[0]
        )
        output.append(prediction)
        window = np.append(window[1:], prediction)
    values = scaler.inverse_transform(np.asarray(output).reshape(-1, 1)).flatten()
    return np.maximum(values, 0)
