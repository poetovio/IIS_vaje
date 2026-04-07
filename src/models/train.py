from pathlib import Path
import os
import random
import joblib
import yaml

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from preprocess import DatePreprocessor, SlidingWindowTransformer


def set_seeds(random_state: int):
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def prepare_station_dataframe(df: pd.DataFrame, target_col: str):
    df = df[["date_to", target_col]].copy()
    date_preprocessor = DatePreprocessor("date_to")
    df = date_preprocessor.fit_transform(df)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.drop(columns=["date_to"], axis=1)
    return df


def create_datasets(df_train, df_test, target_col, window_size):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    train_scaled = numeric_pipeline.fit_transform(df_train[[target_col]])
    test_scaled = numeric_pipeline.transform(df_test[[target_col]])

    sliding_window_transformer = SlidingWindowTransformer(window_size)

    X_train, y_train = sliding_window_transformer.transform(train_scaled)
    X_test, y_test = sliding_window_transformer.transform(test_scaled)

    return X_train, y_train, X_test, y_test, numeric_pipeline, sliding_window_transformer


def evaluate_predictions(y_true_scaled, y_pred_scaled, scaler):
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def export_model_to_onnx(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(output_path), format="onnx")


def train_for_station(station_path: Path, models_dir: Path, params: dict):
    station = station_path.stem
    target_col = params["target_col"]
    test_size = params["test_size"]
    window_size = params["window_size"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    min_rows = params["min_rows"]

    df = pd.read_csv(station_path)
    df = prepare_station_dataframe(df, target_col)

    non_null_count = df[target_col].notna().sum()

    if len(df) < min_rows:
        print(f"Skipping {station}: premalo vrstic ({len(df)}).")
        return None

    if non_null_count == 0:
        print(f"Skipping {station}: stolpec {target_col} je v celoti prazen.")
        return None

    if non_null_count < window_size + 10:
        print(f"Skipping {station}: premalo dejanskih meritev v stolpcu {target_col} ({non_null_count}).")
        return None

    if len(df) <= test_size + window_size:
        print(f"Skipping {station}: premalo podatkov za train/test split.")
        return None

    df_test = df.iloc[-test_size:].copy()
    df_train = df.iloc[:-test_size].copy()

    train_non_null = df_train[target_col].notna().sum()
    test_non_null = df_test[target_col].notna().sum()

    if train_non_null == 0:
        print(f"Skipping {station}: train del nima nobene veljavne vrednosti za {target_col}.")
        return None

    if test_non_null == 0:
        print(f"Skipping {station}: test del nima nobene veljavne vrednosti za {target_col}.")
        return None

    X_train, y_train, X_test, y_test, numeric_pipeline, sliding_window_transformer = create_datasets(
        df_train=df_train,
        df_test=df_test,
        target_col=target_col,
        window_size=window_size,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping {station}: sliding window ni vrnil dovolj podatkov.")
        return None

    print(f"{station} -> X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"{station} -> X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    with mlflow.start_run(run_name=f"train_{station}", nested=True):
        mlflow.log_param("station", station)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("min_rows", min_rows)

        model = build_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        mlflow.tensorflow.autolog(log_models=False)

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )

        y_pred_test = model.predict(X_test, verbose=0)
        test_mae, test_mse, test_rmse = evaluate_predictions(
            y_true_scaled=y_test,
            y_pred_scaled=y_pred_test,
            scaler=numeric_pipeline.named_steps["scaler"],
        )

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)

        print(f"{station} -> Test MAE: {test_mae}")
        print(f"{station} -> Test MSE: {test_mse}")
        print(f"{station} -> Test RMSE: {test_rmse}")

        full_non_null = df[target_col].notna().sum()
        if full_non_null == 0:
            print(f"Skipping full retrain for {station}: celoten dataset nima veljavnih vrednosti.")
            return None

        full_scaled = numeric_pipeline.fit_transform(df[[target_col]])
        X_full, y_full = sliding_window_transformer.transform(full_scaled)

        if len(X_full) == 0:
            print(f"Skipping full retrain for {station}: premalo podatkov.")
            return None

        model_full = build_model((X_full.shape[1], X_full.shape[2]))

        model_full.fit(
            X_full,
            y_full,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )

        y_pred_full = model_full.predict(X_full, verbose=0)
        full_mae, full_mse, full_rmse = evaluate_predictions(
            y_true_scaled=y_full,
            y_pred_scaled=y_pred_full,
            scaler=numeric_pipeline.named_steps["scaler"],
        )

        mlflow.log_metric("full_mae", full_mae)
        mlflow.log_metric("full_mse", full_mse)
        mlflow.log_metric("full_rmse", full_rmse)

        print(f"{station} -> Full dataset MAE: {full_mae}")
        print(f"{station} -> Full dataset MSE: {full_mse}")
        print(f"{station} -> Full dataset RMSE: {full_rmse}")

        models_dir.mkdir(parents=True, exist_ok=True)

        keras_path = models_dir / f"model_{station}.keras"
        onnx_path = models_dir / f"model_{station}.onnx"
        pipeline_path = models_dir / f"pipeline_{station}.pkl"

        model_full.save(keras_path)
        export_model_to_onnx(model_full, onnx_path)

        pipeline_artifact = {
            "numeric_pipeline": numeric_pipeline,
            "sliding_window_transformer": sliding_window_transformer,
            "target_col": target_col,
            "window_size": window_size,
        }

        joblib.dump(pipeline_artifact, pipeline_path)

        mlflow.log_artifact(str(keras_path))
        mlflow.log_artifact(str(onnx_path))
        mlflow.log_artifact(str(pipeline_path))

        return {
            "station": station,
            "rows": len(df),
            "non_null_count": non_null_count,
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "full_mae": full_mae,
            "full_mse": full_mse,
            "full_rmse": full_rmse,
        }


def main():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    random_state = params["random_state"]
    set_seeds(random_state)

    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    mlflow.set_experiment(params["mlflow_experiment"])

    data_dir = project_root / "data" / "preprocessed" / "air"
    models_dir = project_root / "models"

    station_files = sorted(data_dir.glob("*.csv"))

    if not station_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    results = []

    with mlflow.start_run(run_name="train_all_stations"):
        mlflow.log_param("target_col", params["target_col"])
        mlflow.log_param("test_size", params["test_size"])
        mlflow.log_param("window_size", params["window_size"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("epochs", params["epochs"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("min_rows", params["min_rows"])

        for station_path in station_files:
            result = train_for_station(
                station_path=station_path,
                models_dir=models_dir,
                params=params,
            )

            if result is not None:
                results.append(result)

        if not results:
            raise RuntimeError("Noben model ni bil naučen.")

        results_df = pd.DataFrame(results)
        metrics_path = models_dir / "training_metrics.csv"
        results_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(str(metrics_path))

        print(results_df)


if __name__ == "__main__":
    main()