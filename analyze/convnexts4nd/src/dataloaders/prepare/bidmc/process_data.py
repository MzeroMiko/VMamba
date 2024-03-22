import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sktime
from sktime.datasets import load_from_tsfile_to_dataframe
import data_loader as data

DATA_PATH = "data/"


def split_data(
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, shuffle=True, seed=0
):
    if shuffle:
        X_all = pd.concat((X_train_orig, X_test_orig))
        y_all = np.concatenate((y_train_orig, y_test_orig))
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_all, y_all, test_size=0.3, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_eval, y_eval, test_size=0.5, random_state=seed + 1
        )
    else:
        X_test, y_test = X_test_orig, y_test_orig
        val_size = int(X_train_orig.shape[0] / 7.0)  # .6 / .1 / .3 split
        X_train, y_train = X_train_orig[:-val_size], y_train_orig[:-val_size]
        X_val, y_val = X_train_orig[-val_size:], y_train_orig[-val_size:]
        # X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.20, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _to_numpy(X):
    """ Convert DataFrame of series into numpy array """
    return np.stack([np.stack(x) for x in X.to_numpy()]).swapaxes(-1, -2)


def process_data(DATASET, shuffle=True, seed=0):
    X_train_orig, y_train_orig = data.load_from_tsfile_to_dataframe(
        os.path.join(f"{DATASET}/BIDMC32{DATASET}_TRAIN.ts"),
        replace_missing_vals_with="NaN",
    )

    X_test_orig, y_test_orig = data.load_from_tsfile_to_dataframe(
        os.path.join(f"{DATASET}/BIDMC32{DATASET}_TEST.ts"),
        replace_missing_vals_with="NaN",
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X_train_orig, y_train_orig, X_test_orig, y_test_orig, shuffle=shuffle, seed=seed
    )

    split = "reshuffle" if shuffle else "original"
    data_dir = os.path.join(DATASET, split)
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "trainx.npy"), _to_numpy(X_train))
    np.save(os.path.join(data_dir, "trainy.npy"), y_train)
    np.save(os.path.join(data_dir, "validx.npy"), _to_numpy(X_val))
    np.save(os.path.join(data_dir, "validy.npy"), y_val)
    np.save(os.path.join(data_dir, "testx.npy"), _to_numpy(X_test))
    np.save(os.path.join(data_dir, "testy.npy"), y_test)

    for f in ["trainx", "trainy", "validx", "validy", "testx", "testy"]:
        df = np.load(f"{DATASET}/{split}/{f}.npy")
        print(f, df.shape, df.dtype)


if __name__ == "__main__":
    for DATASET in ["RR", "HR", "SpO2"]:
        process_data(DATASET, shuffle=True)
