import os
import pickle
import argparse
import logging

from tqdm import tqdm

import numpy as np
import pandas as pd

from egoviz.models.processing import load_pickle


def _load_mapping_df():
    """Load the mapping dataframe."""
    url = "https://docs.google.com/spreadsheets/d/1X6g7qLgrRNCG3ou6thtGThSjH5U4200l/export?gid=1445909165&format=csv"
    return pd.read_csv(url, index_col=0)


def remap_detic_classes(orig_classes: list[int], mapping_df: pd.DataFrame) -> list[str]:
    """Remap original DETIC classes to functional categories."""
    return [mapping_df.loc[c, "remapped_classes"] for c in orig_classes]


def remapped_class_names_to_ids(remapped_classes: list[str]) -> list[int]:
    """Convert remapped class names to class IDs."""
    class_to_id = {
        "animal": 0,
        "food": 1,
        "plant": 2,
        "sports_equipment": 3,
        "musical_instrument": 4,
        "wheelchair_walker": 5,
        "home_appliance_tool": 6,
        "kitchen_utensils": 7,
        "tableware": 8,
        "drinkware": 9,
        "kitchen_appliance": 10,
        "furniture": 11,
        "cabinetry": 12,
        "furnishing": 13,
        "house_fixtures": 14,
        "electronics": 15,
        "tv_computer": 16,
        "phone_tablet": 17,
        "cleaning_product": 18,
        "toiletries": 19,
        "bathroom_fixture": 20,
        "office_stationary": 21,
        "clothing": 22,
        "hat": 23,
        "footwear": 24,
        "clothing_accessory": 25,
        "bag": 26,
        "other": 27,
        "sink": 28,
        "background": 27,
    }

    return [class_to_id[c] for c in remapped_classes]


def filter_out_detic_class_id(preds: dict, class_id_filter: int) -> dict:
    """Filter out a specific DETIC class from the predictions."""
    indices_to_keep = [
        i for i, class_id in enumerate(preds["classes"]) if class_id != class_id_filter
    ]

    preds["boxes"] = [preds["boxes"][i] for i in indices_to_keep]
    preds["scores"] = [preds["scores"][i] for i in indices_to_keep]
    preds["classes"] = [preds["classes"][i] for i in indices_to_keep]
    preds["metadata"] = [preds["metadata"][i] for i in indices_to_keep]

    return preds


def process_detic_preds(preds: dict, mapping_df) -> dict:
    """Process single pkl file of predictions."""

    # filter out human class (792)
    preds = filter_out_detic_class_id(preds, 792)

    # convert boxes to int
    preds["boxes"] = np.array(preds["boxes"], dtype=int)

    # remap classes
    preds["remapped_metadata"] = remap_detic_classes(preds["classes"], mapping_df)

    # convert remapped classes to class IDs
    preds["remapped_classes"] = remapped_class_names_to_ids(preds["remapped_metadata"])

    return preds


def process_detic_data(dirpath: str):
    """Process DETIC data."""

    mapping_df = _load_mapping_df()

    for file in tqdm(os.listdir(dirpath)):
        if file.endswith(".pkl"):
            filepath = os.path.join(dirpath, file)
            preds = load_pickle(filepath)
            preds = process_detic_preds(preds, mapping_df)

            # save processed predictions
            with open(filepath, "wb") as f:
                pickle.dump(preds, f)

            logging.info(f"Processed predictions saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Process raw detic preds")
    parser.add_argument("dirpath", help="Folder containing pkl files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    process_detic_data(args.dirpath)


if __name__ == "__main__":
    main()
