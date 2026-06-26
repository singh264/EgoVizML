"""
This script is modified from Detic/demo.py.

The original script is used to run inference on a single image.
This script is used to run inference on a directory of images.

Example usage:

```bash
cd Detic

(if you don't want to save output images)
python run_detic.py --input-dir /path/to/input/dir --output-dir /path/to/output/dir --no-images 


(if you want to save output images)
python run_detic.py --input-dir /path/to/input/dir --output-dir /path/to/output/dir
```
"""

import argparse
import glob
import os
import cv2
import tqdm
import pickle
import concurrent.futures
import logging
import time
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
from detectron2.data import MetadataCatalog


def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 inference on images")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output and predictions"
    )
    parser.add_argument(
        "--no-images", required=False, action="store_true", help="Save output images"
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[
            "MODEL.WEIGHTS",
            "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        ],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    return parser


def process_image(input_path, args, demo):
    img = read_image(input_path, format="BGR")
    start_time = time.time()  # Start timing

    predictions, visualized_output, metadata = demo.run_on_image(img)

    # Calculate processing time
    processing_time = time.time() - start_time

    # Log progress information
    if "instances" in predictions:
        num_instances = len(predictions["instances"])
        progress_info = "{} instances detected".format(num_instances)
    else:
        progress_info = "Processing completed"

    logging.info("{}: {} in {:.2f}s".format(input_path, progress_info, processing_time))

    out_filename = os.path.join(args.output_dir, os.path.basename(input_path))

    # Save visualized output (optional)
    if not args.no_images:
        visualized_output.save(out_filename)

    # Save predictions as pkl
    classes = predictions["instances"].pred_classes.cpu().numpy()
    class_names = [metadata.thing_classes[i] for i in classes]

    preds = {
        "boxes": predictions["instances"].pred_boxes.tensor.cpu().numpy(),
        "scores": predictions["instances"].scores.cpu().numpy(),
        "classes": classes,
        "metadata": class_names,
    }

    out_filename_pkl = os.path.splitext(out_filename)[0] + "_detic.pkl"
    with open(out_filename_pkl, "wb") as file:
        pickle.dump(preds, file)


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)

    input_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = [
            executor.submit(process_image, input_path, args, demo)
            for input_path in input_paths
        ]

        # Ensure all tasks are completed
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            pass
