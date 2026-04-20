"""
Integration and unit tests for the egoviz package.

Run with: pytest tests/
"""

import os
import pandas as pd
import pytest

from egoviz.models.processing import (
    load_pickle,
    binary_presence,
    generate_df_from_preds,
    generate_binary_presence_df,
    generate_counts_df,
    row_wise_min_max_scaling,
)
from egoviz.models.inference import load_production_model, predict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_PATH = os.path.join(ROOT, "models", "binary_active_logreg.joblib")
ALL_PREDS_PATH = os.path.join(DATA_DIR, "home_data_all_preds.pkl")
ALL_PREDS_DF_PATH = os.path.join(DATA_DIR, "home_data_all_preds_df.pkl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_preds():
    return load_pickle(ALL_PREDS_PATH)


@pytest.fixture(scope="module")
def preds_df(raw_preds):
    return generate_df_from_preds(raw_preds)


@pytest.fixture(scope="module")
def binary_active_df():
    df = load_pickle(ALL_PREDS_DF_PATH)
    return generate_binary_presence_df(df)


@pytest.fixture(scope="module")
def production_model():
    return load_production_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Unit tests — processing
# ---------------------------------------------------------------------------


def test_binary_presence_deduplicates():
    classes = ["spoon", "plate", "fork", "plate"]
    active = [False, False, False, True]
    out_classes, out_active = binary_presence(classes, active)
    assert out_classes == ["spoon", "plate", "fork"]
    assert out_active == [False, True, False]


def test_binary_presence_all_active():
    classes = ["cup", "cup", "bowl"]
    active = [True, True, False]
    out_classes, out_active = binary_presence(classes, active)
    assert out_classes == ["cup", "bowl"]
    assert out_active == [True, False]


def test_row_wise_min_max_scaling():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 8, 9], "C": [3, 12, 6]})
    df_scaled = row_wise_min_max_scaling(df)
    expected = pd.DataFrame(
        {"A": [0.0, 0.0, 0.0], "B": [0.5, 0.6, 1.0], "C": [1.0, 1.0, 0.5]}
    )
    pd.testing.assert_frame_equal(df_scaled, expected)


def test_row_wise_min_max_scaling_bounds():
    df = pd.DataFrame({"x": [1, 5], "y": [3, 10], "z": [5, 1]})
    df_scaled = row_wise_min_max_scaling(df)
    numeric = df_scaled.select_dtypes(include="number")
    assert (numeric.values >= 0).all()
    assert (numeric.values <= 1).all()


# ---------------------------------------------------------------------------
# Integration tests — pipeline with real data
# ---------------------------------------------------------------------------


def test_load_pickle_all_preds(raw_preds):
    assert isinstance(raw_preds, dict)
    assert len(raw_preds) > 0
    first = next(iter(raw_preds.values()))
    assert "remapped_metadata" in first
    assert "active_objects" in first


def test_generate_df_from_preds(preds_df):
    assert isinstance(preds_df, pd.DataFrame)
    assert {"video", "frame", "classes", "active", "adl"}.issubset(preds_df.columns)
    assert len(preds_df) > 0


def test_generate_counts_df(preds_df):
    counts_df = generate_counts_df(preds_df)
    assert isinstance(counts_df, pd.DataFrame)
    assert "video" in counts_df.columns
    assert "adl" in counts_df.columns
    assert any(c.startswith("count_") for c in counts_df.columns)


def test_generate_binary_presence_df(binary_active_df):
    assert isinstance(binary_active_df, pd.DataFrame)
    assert "video" in binary_active_df.columns
    assert "adl" in binary_active_df.columns
    assert any(c.startswith("active_") for c in binary_active_df.columns)


def test_row_wise_scaling_on_real_data(binary_active_df):
    scaled = row_wise_min_max_scaling(binary_active_df)
    numeric = scaled.select_dtypes(include="number")
    assert (numeric.values >= 0).all()
    assert (numeric.values <= 1).all()


# ---------------------------------------------------------------------------
# Integration tests — model inference
# ---------------------------------------------------------------------------


def test_load_production_model(production_model):
    assert production_model is not None
    assert hasattr(production_model, "model")
    assert hasattr(production_model, "label_encoder")
    assert hasattr(production_model, "feature_names")
    assert hasattr(production_model, "validation_performance")


def test_predict(binary_active_df, production_model):
    import polars as pl

    scaled = row_wise_min_max_scaling(binary_active_df)

    # Fill any features the model expects that aren't in the data
    for col in production_model.feature_names:
        if col not in scaled.columns:
            scaled[col] = 0.0

    scaled_pl = pl.from_pandas(scaled)
    results = predict(scaled_pl, production_model)

    assert "predicted_label" in results.columns
    assert len(results) == len(scaled)
    assert any(c.startswith("prob_") for c in results.columns)
