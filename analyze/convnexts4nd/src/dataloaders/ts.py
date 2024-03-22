"""Time series datasets, especially for medical time series."""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dataloaders.base import default_data_path, SequenceDataset, deprecated

class BIDMC(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc"
    d_input = 2

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def l_output(self):
        return 4000 if self.prediction else 0

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        if self.prediction:
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"

class EegDataset(SequenceDataset):

    _name_ = "eegseizure"

    init_defaults = {
        "l_output": 0,
        "d_input": 19,
        "d_output": 2,
        "machine": "gemini",
        "hospital": "stanford",
        "clip_len": 60,
        "stride": 60,
        "offset": 0,
        "ss_clip_len": 0,
        "use_age": False,
        "gnn": False,
        "fft": False,
        "rerun_meerkatdp": False,
        "streaming_eval": False,
        "sz_label_sensitivity": 1,
    }

    def setup(self):
        import meerkat as mk
        from meerkat.contrib.eeg import (build_stanford_eeg_dp,
                                         build_streaming_stanford_eeg_dp,
                                         build_tuh_eeg_dp)
        from torch.utils.data import WeightedRandomSampler

        assert self.sz_label_sensitivity <= self.clip_len

        # from src.dataloaders.eegseizure import balance_dp, split_dp, merge_in_split
        if self.machine == "gemini":
            data_dir = "/media/4tb_hdd"
            data_dir_tuh = "/media/nvme_data/siyitang/TUH_eeg_seq_v1.5.2"
            raw_tuh_data_dir = "/media/nvme_data/TUH/v1.5.2"
        elif self.machine == "zaman":
            data_dir = "/data/ssd1crypt/datasets"
            data_dir_tuh = "/data/ssd1crypt/datasets/TUH_v1.5.2"
            raw_tuh_data_dir = data_dir_tuh

        if self.hospital == "tuh":
            dp = build_tuh_eeg_dp(
                f"{data_dir_tuh}/resampled_signal",
                f"{raw_tuh_data_dir}/edf",
                clip_len=self.clip_len,
                offset=self.offset,
                ss_clip_len=self.ss_clip_len,
                gnn=self.gnn,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

        else:
            dp = build_stanford_eeg_dp(
                f"{data_dir}/eeg_data/stanford/stanford_mini",
                f"{data_dir}/eeg_data/lpch/lpch",
                "/home/ksaab/Documents/meerkat/meerkat/contrib/eeg/file_markers",
                clip_len=self.clip_len,
                offset=self.offset,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

        if self.streaming_eval:
            streaming_dp = build_streaming_stanford_eeg_dp(
                f"{data_dir}/SEC-0.1/stanford",
                f"{data_dir}/SEC-0.1/lpch",
                "/data/crypt/eegdbs/SEC-0.1/SEC-0.1-sz-annotations-match-lvis",
                clip_len=self.clip_len,
                stride=self.stride,
                sz_label_sensitivity=self.sz_label_sensitivity,
                train_frac=0.0,
                valid_frac=0.5,
                test_frac=0.5,
                skip_terra_cache=self.rerun_meerkatdp,
            ).load()

            # remove patients in dp that are in streaming_dp
            streaming_patients = streaming_dp["patient_id"].unique()
            keep_patient_mask = np.array(
                [patient not in streaming_patients for patient in dp["patient_id"]]
            )
            dp = dp.lz[keep_patient_mask]

        # shuffle datapanel
        np.random.seed(0)
        ndxs = np.arange(len(dp))
        np.random.shuffle(ndxs)
        dp = dp.lz[ndxs]

        val_split = "valid"
        test_split = "test"

        input_key = "input"
        target_key = "target"

        train_mask = dp["split"] == "train"
        val_mask = dp["split"] == val_split
        test_mask = dp["split"] == test_split

        if self.fft:
            input_key = "fft_input"
            self.d_input = 1900
        if self.ss_clip_len > 0:
            target_key = "ss_output"
            self.d_output = 19*100 #int(19 * (200* self.ss_clip_len / 2))
            self.l_output = self.ss_clip_len

            # train_mask = np.logical_and(train_mask.data,(dp["target"]==1).data)
            # val_mask = np.logical_and(val_mask.data,(dp["target"]==1).data)
            # test_mask = np.logical_and(test_mask.data,(dp["target"]==1).data)

        self.dataset_train = dp.lz[train_mask][
            input_key, target_key, "age", "target"
        ]
        self.dataset_val = dp.lz[val_mask][
            input_key, target_key, "age", "target"
        ]
        self.dataset_test = dp.lz[test_mask][
            input_key, target_key, "age"
        ]


        # define whats returned by datasets
        if self.gnn:
            lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key]).to(torch.long),
                x[input_key][1],  # graph supports
            )
            if self.ss_clip_len > 0:
                lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key][0]).to(torch.long),
                torch.tensor(x[target_key][0]).to(torch.long), # decoder takes y as well
                x[input_key][1],  # graph supports
            )
            if self.use_age:
                lambda_fnc = lambda x: (
                    x[input_key][0],
                    torch.tensor(x[target_key]).to(torch.long),
                    x[input_key][1],  # graph supports
                    torch.tensor(x["age"]).to(torch.float),
                )
        else:
            lambda_fnc = lambda x: (
                x[input_key][0],
                torch.tensor(x[target_key]).to(torch.long)
                if self.ss_clip_len == 0
                else x[target_key],
            )
            if self.use_age:
                lambda_fnc = lambda x: (
                    x[input_key][0],
                    torch.tensor(x[target_key]).to(torch.long)
                    if self.ss_clip_len == 0
                    else x[target_key],
                    torch.tensor(x["age"]).to(torch.float),
                )

        self.dataset_train["examples"] = mk.LambdaColumn(self.dataset_train, lambda_fnc)

        if self.ss_clip_len == 0:
            # define train sampler
            train_target = self.dataset_train["target"].data.astype(np.int)
            class_sample_count = np.array(
                [len(np.where(train_target == t)[0]) for t in np.unique(train_target)]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in train_target])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
        else:
            samples_weight = torch.ones(len(self.dataset_train))
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        self.dataset_val["examples"] = mk.LambdaColumn(self.dataset_val, lambda_fnc)

        self.dataset_test["examples"] = mk.LambdaColumn(self.dataset_test, lambda_fnc)
        print(
            f"Train:{len(self.dataset_train)} Validation:{len(self.dataset_val)} Test:{len(self.dataset_test)}"
        )

        if self.streaming_eval:
            self.stream_dataset_val = streaming_dp.lz[streaming_dp["split"] == "valid"][
                input_key, "target", "age", "clip_start"
            ]
            self.stream_dataset_test = streaming_dp.lz[streaming_dp["split"] == "test"][
                input_key, "target", "age", "clip_start"
            ]

            self.stream_dataset_val["examples"] = mk.LambdaColumn(
                self.stream_dataset_val,
                lambda x: (
                    x[input_key],
                    torch.tensor(x["target"]).to(torch.long),
                    torch.tensor(x["age"]).to(torch.float),
                    torch.tensor(x["clip_start"]).to(torch.float),
                ),
            )

            self.stream_dataset_test["examples"] = mk.LambdaColumn(
                self.stream_dataset_test,
                lambda x: (
                    x[input_key],
                    torch.tensor(x["target"]).to(torch.long),
                    torch.tensor(x["age"]).to(torch.float),
                    torch.tensor(x["clip_start"]).to(torch.float),
                ),
            )

    def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing

        return torch.utils.data.DataLoader(
            self.dataset_train["examples"],
            sampler=self.train_sampler,
            **kwargs,
        )

    def val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing
        return torch.utils.data.DataLoader(
            self.dataset_val["examples"],
            **kwargs,
        )

    def test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        # No collate_fn is passed in: the default one does the right thing
        return torch.utils.data.DataLoader(
            self.dataset_test["examples"],
            **kwargs,
        )

    def stream_val_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if self.streaming_eval:
            # No collate_fn is passed in: the default one does the right thing
            return torch.utils.data.DataLoader(
                self.stream_dataset_val["examples"],
                **kwargs,
            )

    def stream_test_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if self.streaming_eval:
            # No collate_fn is passed in: the default one does the right thing
            return torch.utils.data.DataLoader(
                self.stream_dataset_test["examples"],
                **kwargs,
            )

class PTBXL(SequenceDataset):

    _name_ = "ptbxl"

    init_defaults = {
        "sampling_rate": 100,
        "duration": 10,
        "nleads": 12,
        "ctype": "superdiagnostic",
        "min_samples": 0,
    }

    @property
    def d_input(self):
        return self.nleads

    def load_raw_data(self, df):
        import wfdb

        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(str(self.data_dir / f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(str(self.data_dir / f)) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.L = self.sampling_rate * self.duration
        self.l_output = 0  # TODO(Priya): This changes with every multilabel setting?

        # PTBXL imports
        import ast

        import pandas as pd
        from sklearn import preprocessing

        # load and convert annotation data
        Y = pd.read_csv(self.data_dir / "ptbxl_database.csv", index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.data_dir / "scp_statements.csv", index_col=0)

        if self.ctype in [
            "diagnostic",
            "subdiagnostic",
            "superdiagnostic",
            "superdiagnostic_multiclass",
        ]:
            agg_df = agg_df[agg_df.diagnostic == 1]

            def aggregate_superdiagnostic_multiclass(y_dic):
                lhmax = -1  # Superclass has the highest likelihood
                superclass = ""
                for key in y_dic.keys():
                    if key in agg_df.index and y_dic[key] > lhmax:
                        lhmax = y_dic[key]
                        superclass = agg_df.loc[key].diagnostic_class
                return superclass

            def aggregate_all_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        tmp.append(key)
                return list(set(tmp))

            def aggregate_subdiagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = agg_df.loc[key].diagnostic_subclass
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            def aggregate_superdiagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = agg_df.loc[key].diagnostic_class
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            # Apply aggregation
            if self.ctype == "superdiagnostic_multiclass":
                Y["target"] = Y.scp_codes.apply(aggregate_superdiagnostic_multiclass)
            elif self.ctype == "subdiagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_subdiagnostic)
            elif self.ctype == "superdiagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_superdiagnostic)
            elif self.ctype == "diagnostic":
                Y["target"] = Y.scp_codes.apply(aggregate_all_diagnostic)

        elif self.ctype in ["form", "rhythm"]:

            if self.ctype == "form":
                agg_df = agg_df[agg_df.form == 1]
            else:
                agg_df = agg_df[agg_df.rhythm == 1]

            def aggregate_form_rhythm(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        c = key
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            Y["target"] = Y.scp_codes.apply(aggregate_form_rhythm)

        elif self.ctype == "all":
            Y["target"] = Y.scp_codes.apply(lambda x: list(set(x.keys())))

        counts = pd.Series(np.concatenate(Y.target.values)).value_counts()
        counts = counts[counts > self.min_samples]
        Y.target = Y.target.apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )
        Y["target_len"] = Y.target.apply(lambda x: len(x))
        Y = Y[Y.target_len > 0]
        # Load raw signal data
        X = self.load_raw_data(Y)

        # Split data into train, val and test
        val_fold = 9
        test_fold = 10

        # Convert labels to multiclass or multilabel targets
        if self.ctype == "superdiagnostic_multiclass":
            le = preprocessing.LabelEncoder()
        else:
            le = preprocessing.MultiLabelBinarizer()

        le.fit(Y.target)
        y = le.transform(Y.target)
        self.d_output = len(le.classes_)

        # Train
        X_train = X[np.where((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))]
        y_train = y[np.where((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))]
        # Val
        X_val = X[np.where(Y.strat_fold == val_fold)]
        y_val = y[np.where(Y.strat_fold == val_fold)]

        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = y[np.where(Y.strat_fold == test_fold)]

        def preprocess_signals(X_train, X_validation, X_test):
            # Standardize data such that mean 0 and variance 1
            ss = preprocessing.StandardScaler()
            ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

            return (
                apply_standardizer(X_train, ss),
                apply_standardizer(X_validation, ss),
                apply_standardizer(X_test, ss),
            )

        def apply_standardizer(X, ss):
            X_tmp = []
            for x in X:
                x_shape = x.shape
                X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
            X_tmp = np.array(X_tmp)
            return X_tmp

        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.tensor(X_train).to(torch.float), torch.tensor(y_train)
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.tensor(X_val).to(torch.float), torch.tensor(y_val)
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.tensor(X_test).to(torch.float), torch.tensor(y_test)
        )

        print(
            f"Train:{len(X_train)} Validation:{len(X_val)} Test:{len(X_test)} Num_classes:{self.d_output}"
        )

        self.collate_fn = None

class IMU(SequenceDataset):
    """IMU (Inertial Measurement Units) dataset from an experimental study on Parkinson patients"""

    _name_ = "imu"
    d_input = 36  # len(imu_config)
    l_output = 0

    @property
    def d_output(self):
        return d_input if self.prediction else 2

    @property
    def init_defaults(self):
        return {
            #'target': 'RR', # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.collate_fn = None

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)

        # dictionary of config name to list of features
        # choose sensors06_chest_lumbar_ankles_feet by default
        # ignore this now as we're only using a fixed set of features
        with open(self.data_dir / "sensor_configs.pkl", "rb") as config_f:
            imu_config_map = pickle.load(config_f)
        imu_config = imu_config_map["sensors06_chest_lumbar_ankles_feet"]

        with open(self.data_dir / "0_train_matrices.pkl", "rb") as f_handle:
            tr = pickle.load(f_handle)
        with open(self.data_dir / "0_val_matrices.pkl", "rb") as f_handle:
            val = pickle.load(f_handle)
        with open(self.data_dir / "0_test_matrices.pkl", "rb") as f_handle:
            te = pickle.load(f_handle)

        X_train = tr[0]
        y_train = tr[1].astype(int)
        X_val = val[0]
        y_val = val[1].astype(int)
        X_test = te[0]
        y_test = te[1].astype(int)

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.tensor(y_train, dtype=torch.long)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.tensor(y_val, dtype=torch.long)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.tensor(y_test, dtype=torch.long)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"IMU_{split}"

