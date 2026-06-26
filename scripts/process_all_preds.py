"""
This script combines the processed and remapped predictions of the detic model into one
file. The predictions are saved in the following format:

all_preds[{adl}_{video}_frame{#}] = {preds for frame #}
"""

import argparse
import logging
import os
import pickle

import numpy as np
import torch
from process_detic import _load_mapping_df, process_detic_preds
from torchvision.ops import box_iou

from egoviz.models.processing import load_pickle
from egoviz.egomodelkit_progress import emit_progress


def get_active_objects(
    detic_boxes: list, shan_boxes: list, active_iou: float = 0.75
) -> list:
    """Get the active objects from the shan preds."""

    if shan_boxes is not None:
        if len(detic_boxes) == 0:
            return np.array([])

        detic_boxes = np.array(detic_boxes).astype(int)
        shan_boxes = np.array([obj[0:4] for obj in shan_boxes]).astype(int)
        ious = box_iou(torch.tensor(detic_boxes), torch.tensor(shan_boxes))
        active = [any(ious[i] >= active_iou) for i in range(len(detic_boxes))]
        return active
    elif len(detic_boxes) > 0 and shan_boxes is None:
        return np.array([False] * len(detic_boxes))
    else:
        return np.array([])


def collect_prediction_pairs(dirpath: str, adls: list) -> list:
    """ Collect matched Detic/Shan prediction files across ADL folders. """
    prediction_pairs = []

    for adl in adls:
        logging.info(f"Processing {adl}...")

        detic_dirpath = os.path.join(dirpath, adl, "detic")
        shan_dirpath = os.path.join(dirpath, adl, "shan")

        if not os.path.exists(detic_dirpath) or not os.path.exists(shan_dirpath):
            continue

        detic_pkl_files = sorted(
            f for f in os.listdir(detic_dirpath) if f.endswith(".pkl")
        )
        
        shan_pkl_files = sorted(
            f for f in os.listdir(shan_dirpath) if f.endswith(".pkl")
        )

        logging.info(f"First 5 files for {adl} detic:")
        logging.info(detic_pkl_files[:5])
        logging.info(f"First 5 files for {adl} shan:")
        logging.info(shan_pkl_files[:5])

        for detic_file, shan_file in zip(detic_pkl_files, shan_pkl_files):
            prediction_pairs.append(
                (
                    adl,
                    detic_dirpath,
                    shan_dirpath,
                    detic_file,
                    shan_file,
                )
            )

    return prediction_pairs


def process_all_preds(dirpath: str, active_iou: float) -> dict:
    """
    The pipeline expects object detection and hand-object interaction
    predictions organized in this structure:

    root_directory/
    ├── communication-management/
    │   ├── detic/
    │   │   ├── video1_frame1.pkl
    │   │   └── ...
    │   └── shan/
    │       ├── video1_frame1.pkl
    │       └── ...
    ├── functional-mobility/
    │   ├── detic/
    │   └── shan/
    └── ...other activity folders... (specified in adls dict below)

    """

    all_preds = {}
    adls = [
        "communication-management",
        "functional-mobility",
        "grooming-health-management",
        "home-management",
        "leisure-other-activities",
        "meal-preparation-cleanup",
        "self-feeding",
        # add any subfolders here
    ]

    prediction_pairs = collect_prediction_pairs(dirpath, adls)
    total_frames = len(prediction_pairs)
    mapping_df = _load_mapping_df()

    emit_progress(
        "adl_prediction_frames_discovered",
        current = 0,
        total = total_frames,
    )

    for processed_frames, (
        adl,
        detic_dirpath,
        shan_dirpath,
        detic_file,
        shan_file,
    ) in enumerate(prediction_pairs, start=1):
        detic_filepath = os.path.join(detic_dirpath, detic_file)
        shan_filepath = os.path.join(shan_dirpath, shan_file)

        detic_preds = load_pickle(detic_filepath)
        shan_preds = load_pickle(shan_filepath)

        # process detic preds
        detic_preds = process_detic_preds(detic_preds, mapping_df)

        # get active objects from shan preds
        shan_boxes = shan_preds["objects"] if shan_preds is not None else None
        detic_boxes = detic_preds["boxes"]
        active_objects = get_active_objects(detic_boxes, shan_boxes, active_iou)

        # add active objects to detic preds
        detic_preds["active_objects"] = active_objects

        # get video name
        video = detic_file.split("_")[0] + "_" + detic_file.split("_")[1]

        # add to all_preds
        all_preds[f"{adl}_{video}"] = detic_preds

        emit_progress(
            "adl_prediction_frame_processed",
            current = processed_frames,
            total = total_frames,
            adl = adl,
            frame = detic_file,
        )

    # save all_preds
    savepath = os.path.join(dirpath, "all_preds.pkl")
    with open(savepath, "wb") as f:
        pickle.dump(all_preds, f)

    return all_preds


def main():
    parser = argparse.ArgumentParser(description="Combine processed detic preds")
    parser.add_argument(
        "dirpath", help="Folder containing pkl files", default="/d/PhD/adl-recognition"
    )
    parser.add_argument(
        "--active_iou",
        help="IoU threshold for determining active objects",
        default=0.75,
        type=float,
    )

    args = parser.parse_args()

    emit_progress("adl_predictions_combining")

    print(f"Processing and computing active objects in {args.dirpath}...")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    process_all_preds(args.dirpath, args.active_iou)

    emit_progress("adl_predictions_combined")


if __name__ == "__main__":
    main()
