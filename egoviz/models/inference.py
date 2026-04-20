import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


@dataclass
class ProductionModel:
    """Container for the production model and its metadata."""

    model: Any
    label_encoder: Any
    feature_names: list
    validation_performance: dict
    preprocessing_params: dict


def train_final_model(
    df_binary_active: pl.DataFrame, save_path: str
) -> ProductionModel:
    """
    Train the final production model using all available data with the optimal configuration
    (Binary + Active features with Logistic Regression).

    Args:
        df_binary_active: Polars DataFrame with binary and active features
        save_path: Path to save the model
    """
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Prepare the data
    feature_cols = [
        col for col in df_binary_active.columns if col not in ["adl", "video"]
    ]

    # Convert to numpy arrays for sklearn
    X = df_binary_active.select(feature_cols).to_numpy()
    y = df_binary_active.select("adl").to_numpy().ravel()

    # Initialize and fit label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Initialize and train the model with optimal parameters
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    # Train on all data
    clf.fit(X, y_encoded)

    # Store validation performance metrics from LOGOCV
    validation_performance = {
        "mean_f1": 0.784882,
        "std_f1": 0.119884,
        "median_f1": 0.811589,
        "auc": 0.94,
        "pct_above_0.5_f1": 1.00,
        "validation_method": "leave-one-subject-out CV",
    }

    # Store preprocessing parameters
    preprocessing_params = {
        "feature_type": "binary_and_active",
        "scaling": "row_wise_min_max",
    }

    # Create ProductionModel instance
    production_model = ProductionModel(
        model=clf,
        label_encoder=label_encoder,
        feature_names=feature_cols,
        validation_performance=validation_performance,
        preprocessing_params=preprocessing_params,
    )

    # Save the model
    joblib.dump(production_model, save_path)
    logging.info(f"Saved production model to {save_path}")

    return production_model


def predict(new_data: pl.DataFrame, production_model: ProductionModel) -> pl.DataFrame:
    """Make predictions on new data using the production model."""
    # Verify features match
    missing_features = set(production_model.feature_names) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing features in new data: {missing_features}")

    # Prepare features in correct order
    X_new = new_data.select(production_model.feature_names).to_numpy()

    # Make predictions
    y_pred = production_model.model.predict(X_new)
    y_pred_labels = production_model.label_encoder.inverse_transform(y_pred)

    # Get probability predictions
    y_prob = production_model.model.predict_proba(X_new)

    # Create probability columns
    prob_data = {
        f"prob_{label}": y_prob[:, i]
        for i, label in enumerate(production_model.label_encoder.classes_)
    }

    # Create results DataFrame
    results = new_data.clone()

    # Add predictions
    results = results.with_columns(
        [
            pl.Series(name="predicted_label", values=y_pred_labels),
            pl.Series(name="predicted_class", values=y_pred),
        ]
    )

    # Add probability columns
    for col_name, probs in prob_data.items():
        results = results.with_columns(pl.Series(name=col_name, values=probs))

    return results


def load_production_model(model_path: str) -> ProductionModel:
    """Load the production model and its metadata."""
    return joblib.load(model_path)
