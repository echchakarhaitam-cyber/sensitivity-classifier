import numpy as np
import pytest

from sensclf.predictions import read_predictions, write_predictions


def test_roundtrip_keeps_nan_and_reorders(tmp_path):
    S = np.array([[0.1, np.nan, 2.0, 0, 0, 0], [0.9, np.nan, -1.5, 1, 1, 1]])
    path = tmp_path / "p.jsonl"
    write_predictions(path, ["a", "b"], S)
    back = read_predictions(path, ["b", "a"])
    np.testing.assert_array_equal(back, S[::-1])


def test_mismatched_documents_raise(tmp_path):
    path = tmp_path / "p.jsonl"
    write_predictions(path, ["a"], np.zeros((1, 6)))
    with pytest.raises(ValueError):
        read_predictions(path, ["a", "b"])
    with pytest.raises(ValueError):
        write_predictions(path, ["a", "b"], np.zeros((1, 6)))
