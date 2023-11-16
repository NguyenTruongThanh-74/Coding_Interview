import sys

sys.path.append("../")
import os
import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, GRU
from data_processing.data_processing_helper import preprocess_data, scale_data, split_data
from config.variables import result_path, feature_file_list, features_path, EPOCHS, BATCH_SIZE
import matplotlib.pyplot as plt
from utils.results_writer import ResultsWriter
import datetime


# Code
def build_rgu_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(GRU(50, input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_absolute_error'])

    # Training the models:
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    return model


def rgu_model():
    for file_name in feature_file_list:
        print(datetime.datetime.now(), " File {}".format(file_name))
        results_writer = ResultsWriter(result_path, file_name.split('.')[0])

        # Read file csv
        dataset_path = os.path.join(features_path, file_name)
        df = pd.read_csv(dataset_path)
        df = preprocess_data(df)
        df = df.set_index('Date_time')

        # Scale the data
        scaler, X_scaler, y_scaler = scale_data(df)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, X_scaler, y_scaler)
        print("The number of training samples: ", X_train.shape[0])
        print("The number of validation samples: ", X_val.shape[0])
        print("The number of testing samples: ", X_test.shape[0])

        print("Starting training...")
        # Train/Predict GRU model
        model = build_rgu_model(X_train, y_train, X_val, y_val)
        print("Testing model")
        ytest_unscaled_prediction = model.predict(X_test)

        # Inverse transform
        ytest_prediction = scaler.inverse_transform(ytest_unscaled_prediction)
        ytest_ground_truth = scaler.inverse_transform(y_test)
        y_train = scaler.inverse_transform(y_train)
        y_val = scaler.inverse_transform(y_val)

        # Write output
        results_writer.write_dl_results(y_train=y_train,
                                        y_val=y_val,
                                        ground_truth=ytest_ground_truth,
                                        predictions=ytest_prediction,
                                        model_name="GRU",
                                        indent=2)


def plot_rgu_chart(result, axis, position):
    ax = axis[position]
    train = result["train"]
    val = result["val"]
    test = result["test"]
    algorithm = result["model"]
    predictions = result["prediction"]

    split_train_index = len(train)
    split_val_index = len(train) + len(val)

    df = pd.DataFrame(train + val + test)
    train_data = df[:split_train_index]
    val_data = df[split_train_index: split_val_index]
    test_data = df[split_val_index:]

    ax.plot(train_data.index, train_data[0], color="blue")
    ax.plot(val_data.index, val_data[0], color="green")
    ax.plot(test_data.index, test_data[0], color="red")
    ax.plot(test_data.index, predictions, color="orange")

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Debts", fontsize=12)

    ax.axvline(val_data.index[0], color='black', ls='--')
    ax.axvline(test_data.index[0], color='black', ls='--')

    ax.legend(["Train values", "Validation values", "Test values", "Predicted values"])
    ax.title.set_text(f"Gru-Dataset ( {result['dataset']} )")


def plot_gru_synthesize_charts():
    date_results_path = "../../results/%s/DL_models/average_results.json"
    figure, axis = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 20))
    plt.figure()
    axis = axis.ravel()
    for idx, file_name in enumerate(['date_smedebtsu.csv', 'lag3_smedebtsu.csv', 'lag3_date_smedebtsu.csv']):
        with open(date_results_path % file_name.replace('.csv', '')) as f:
            results = json.load(f)
            plot_rgu_chart(results, axis, idx)
    plt.show()
