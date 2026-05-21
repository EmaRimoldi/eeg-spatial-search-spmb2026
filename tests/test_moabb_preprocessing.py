import sys
import types

import mne
import numpy as np
import pytest

from src.data import preprocessing


def _make_physionet_like_raw(sfreq: float, seed: int) -> mne.io.RawArray:
    ch_names = preprocessing._get_channel_names("PhysionetMI")
    n_samples = int(round(sfreq * 12.0))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(ch_names), n_samples)).astype(np.float64)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 6.0],
            duration=[0.0, 0.0],
            description=["left_hand", "right_hand"],
        )
    )
    return raw


def _make_physionet_feet_hands_raw(sfreq: float, seed: int) -> mne.io.RawArray:
    ch_names = preprocessing._get_channel_names("PhysionetMI")
    n_samples = int(round(sfreq * 12.0))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(ch_names), n_samples)).astype(np.float64)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 3.0, 6.0, 8.0],
            duration=[0.0, 0.0, 0.0, 0.0],
            description=["feet", "hands", "feet", "rest"],
        )
    )
    return raw


def _make_two_class_raw(dataset_name: str, sfreq: float, seed: int) -> mne.io.RawArray:
    ch_names = preprocessing._get_channel_names(dataset_name)
    n_samples = int(round(sfreq * 12.0))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(ch_names), n_samples)).astype(np.float64)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 6.0],
            duration=[0.0, 0.0],
            description=["left_hand", "right_hand"],
        )
    )
    return raw


def _make_stim_two_class_raw(dataset_name: str, sfreq: float, seed: int) -> mne.io.RawArray:
    ch_names = preprocessing._get_channel_names(dataset_name)
    n_samples = int(round(sfreq * 26.0))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(ch_names) + 1, n_samples)).astype(np.float64)
    stim = np.zeros(n_samples, dtype=np.float64)
    stim[int(round(sfreq * 2.0))] = 1.0
    stim[int(round(sfreq * 14.0))] = 2.0
    data[-1] = stim

    info = mne.create_info(
        ch_names=ch_names + ["Stim"],
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names) + ["stim"],
    )
    return mne.io.RawArray(data, info, verbose=False)


def test_raw_to_epochs_resamples_physionet_to_target_length():
    cfg = preprocessing.DATASET_CONFIGS["PhysionetMI"]
    expected_n_times = preprocessing._expected_n_times(cfg)

    X_128, y_128 = preprocessing._raw_to_epochs(_make_physionet_like_raw(128.0, seed=1), cfg)
    X_160, y_160 = preprocessing._raw_to_epochs(_make_physionet_like_raw(160.0, seed=2), cfg)

    assert X_128 is not None and y_128 is not None
    assert X_160 is not None and y_160 is not None
    assert X_128.shape == (2, 64, expected_n_times)
    assert X_160.shape == (2, 64, expected_n_times)
    assert expected_n_times == 641
    np.testing.assert_array_equal(y_128, np.array([0, 1]))
    np.testing.assert_array_equal(y_160, np.array([0, 1]))


class _FakePhysionetMI:
    subject_list = [100, 101]

    def get_data(self, subjects):
        runs = {}
        for subject in subjects:
            sfreq = 128.0 if subject == 100 else 160.0
            runs[subject] = {"0": {"0": _make_physionet_like_raw(sfreq, seed=subject)}}
        return runs


def test_process_dataset_concatenates_mixed_physionet_sample_rates(monkeypatch):
    fake_datasets = types.ModuleType("moabb.datasets")
    fake_datasets.BNCI2014_001 = _FakePhysionetMI
    fake_datasets.PhysionetMI = _FakePhysionetMI
    fake_moabb = types.ModuleType("moabb")
    fake_moabb.datasets = fake_datasets

    monkeypatch.setitem(sys.modules, "moabb", fake_moabb)
    monkeypatch.setitem(sys.modules, "moabb.datasets", fake_datasets)

    cfg = dict(preprocessing.DATASET_CONFIGS["PhysionetMI"])
    cfg["subjects_train"] = []
    cfg["subjects_val"] = []
    cfg["subjects_test"] = [100, 101]

    _, _, _, _, X_test, y_test = preprocessing._process_dataset(
        "PhysionetMI", cfg, split_names=("test",)
    )

    assert X_test.shape == (4, 64, 641)
    np.testing.assert_array_equal(y_test, np.array([0, 1, 0, 1]))


def test_physionet_hands_and_rest_are_ignored_not_remapped():
    cfg = preprocessing.DATASET_CONFIGS["PhysionetMI"]
    X, y = preprocessing._raw_to_epochs(_make_physionet_feet_hands_raw(160.0, seed=3), cfg)

    assert X is not None and y is not None
    assert X.shape == (2, 64, preprocessing._expected_n_times(cfg))
    np.testing.assert_array_equal(y, np.array([2, 2]))


@pytest.mark.parametrize(
    ("dataset_name", "n_channels"),
    [("BNCI2014_004", 3), ("Cho2017", 64)],
)
def test_raw_to_epochs_supports_new_two_class_moabb_datasets(dataset_name, n_channels):
    cfg = preprocessing.DATASET_CONFIGS[dataset_name]
    X, y = preprocessing._raw_to_epochs(
        _make_two_class_raw(dataset_name, cfg["sfreq_target"], seed=n_channels),
        cfg,
    )

    assert X is not None and y is not None
    assert X.shape == (2, n_channels, preprocessing._expected_n_times(cfg))
    np.testing.assert_array_equal(y, np.array([0, 1]))


def test_raw_to_epochs_supports_stim_channel_moabb_datasets():
    cfg = preprocessing.DATASET_CONFIGS["Shin2017A"]
    X, y = preprocessing._raw_to_epochs(
        _make_stim_two_class_raw("Shin2017A", cfg["sfreq_target"], seed=30),
        cfg,
    )

    assert X is not None and y is not None
    assert X.shape == (2, 30, preprocessing._expected_n_times(cfg))
    np.testing.assert_array_equal(y, np.array([0, 1]))


class _FakeBNCI2014_004:
    subject_list = [1, 2]

    def get_data(self, subjects):
        return {
            subject: {"0": {"0": _make_two_class_raw("BNCI2014_004", 250.0, seed=subject)}}
            for subject in subjects
        }


class _FakeCho2017:
    subject_list = [45, 46]

    def get_data(self, subjects):
        return {
            subject: {"0": {"0": _make_two_class_raw("Cho2017", 512.0, seed=subject)}}
            for subject in subjects
        }


class _FakeShin2017A:
    subject_list = [25, 26]

    def __init__(self, accept=False):
        assert accept is True

    def get_data(self, subjects):
        return {
            subject: {"0imagery": {"0": _make_stim_two_class_raw("Shin2017A", 200.0, seed=subject)}}
            for subject in subjects
        }


@pytest.mark.parametrize(
    ("dataset_name", "fake_cls", "expected_shape"),
    [
        ("BNCI2014_004", _FakeBNCI2014_004, (4, 3, 1001)),
        ("Cho2017", _FakeCho2017, (4, 64, 1537)),
        ("Shin2017A", _FakeShin2017A, (4, 30, 2001)),
    ],
)
def test_process_dataset_dispatches_new_dataset_classes(monkeypatch, dataset_name, fake_cls, expected_shape):
    fake_datasets = types.ModuleType("moabb.datasets")
    setattr(fake_datasets, dataset_name, fake_cls)
    fake_moabb = types.ModuleType("moabb")
    fake_moabb.datasets = fake_datasets

    monkeypatch.setitem(sys.modules, "moabb", fake_moabb)
    monkeypatch.setitem(sys.modules, "moabb.datasets", fake_datasets)

    cfg = dict(preprocessing.DATASET_CONFIGS[dataset_name])
    cfg["subjects_train"] = []
    cfg["subjects_val"] = []
    cfg["subjects_test"] = list(fake_cls.subject_list)

    _, _, _, _, X_test, y_test = preprocessing._process_dataset(
        dataset_name, cfg, split_names=("test",)
    )

    assert X_test.shape == expected_shape
    np.testing.assert_array_equal(y_test, np.array([0, 1, 0, 1]))


def test_process_dataset_can_force_subject_wise_loading(monkeypatch):
    class _FakeSerialDataset:
        subject_list = [1, 2]
        calls = []

        def __init__(self):
            type(self).calls = []

        def get_data(self, subjects):
            type(self).calls.append(tuple(subjects))
            subject = subjects[0]
            return {subject: {"0": {"0": _make_two_class_raw("BNCI2014_004", 250.0, seed=subject)}}}

    fake_datasets = types.ModuleType("moabb.datasets")
    setattr(fake_datasets, "BNCI2014_004", _FakeSerialDataset)
    fake_moabb = types.ModuleType("moabb")
    fake_moabb.datasets = fake_datasets

    monkeypatch.setitem(sys.modules, "moabb", fake_moabb)
    monkeypatch.setitem(sys.modules, "moabb.datasets", fake_datasets)

    cfg = dict(preprocessing.DATASET_CONFIGS["BNCI2014_004"])
    cfg["subjects_train"] = [1, 2]
    cfg["subjects_val"] = []
    cfg["subjects_test"] = []
    cfg["load_subjects_individually"] = True

    X_train, y_train, *_ = preprocessing._process_dataset("BNCI2014_004", cfg, split_names=("train",))

    assert _FakeSerialDataset.calls == [(1,), (2,)]
    assert X_train.shape == (4, 3, 1001)
    np.testing.assert_array_equal(y_train, np.array([0, 1, 0, 1]))


def test_channel_lists_cover_new_dataset_layouts():
    assert preprocessing._get_channel_names("BNCI2014_004") == ["C3", "Cz", "C4"]
    assert len(preprocessing._get_channel_names("Cho2017")) == 64
    assert preprocessing._get_channel_names("Cho2017")[:4] == ["Fp1", "AF7", "AF3", "F1"]
    assert len(preprocessing._get_channel_names("Shin2017A")) == 30
    assert preprocessing._get_channel_names("Shin2017A")[:4] == ["AFp1", "AFp2", "AFF1h", "AFF2h"]


def test_extended_10_5_channels_get_real_coords_from_mne():
    coords = preprocessing._get_mne_coords_3d(["AFF1h", "FCC3h", "PPO2h"])
    assert coords.shape == (3, 3)
    assert not np.allclose(coords, 0.0)


def test_load_moabb_dataset_handles_test_only_splits(monkeypatch, tmp_path):
    expected_n_times = preprocessing._expected_n_times(preprocessing.DATASET_CONFIGS["PhysionetMI"])
    X_test = np.ones((3, 64, expected_n_times), dtype=np.float32)
    y_test = np.array([0, 1, 0], dtype=np.int64)

    monkeypatch.setattr(
        preprocessing,
        "_load_or_process",
        lambda *args, **kwargs: (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            X_test,
            y_test,
        ),
    )
    monkeypatch.setattr(
        preprocessing,
        "_build_channel_metadata",
        lambda dataset_name, cfg, n_channels: {"dataset": dataset_name, "n_channels": n_channels},
    )

    train_loader, val_loader, test_loader, metadata = preprocessing.load_moabb_dataset(
        "PhysionetMI",
        batch_size=2,
        cache_dir=str(tmp_path),
        split_names=("test",),
    )

    assert len(train_loader.dataset) == 0
    assert len(val_loader.dataset) == 0
    assert len(test_loader.dataset) == 3
    assert metadata == {"dataset": "PhysionetMI", "n_channels": 64}
