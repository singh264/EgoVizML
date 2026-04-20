from collections import Counter

import pickle
import pandas as pd


def load_pickle(filepath: str):
    """Loads a pickle file from a filepath."""

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def generate_df_from_preds(preds: dict[dict], save_path: str | None = None):
    """Generates a dataframe from the predictions of the model."""

    df = pd.DataFrame(columns=["video", "frame", "classes", "active", "adl"])

    for idx, dets in preds.items():
        adl = idx.split("_", 1)[0]
        video = idx.split("_")[1]
        frame = idx.split("_")[2]

        assert "remapped_metadata" in dets.keys(), "remapped_metadata not in dets"
        assert "active_objects" in dets.keys(), "active_objects not in dets"

        classes = dets["remapped_metadata"]
        active = dets["active_objects"]

        row = {
            "video": video,
            "frame": frame,
            "classes": classes,
            "adl": adl,
            "active": active,
        }

        df.loc[len(df)] = row

    if save_path:
        df.to_pickle(save_path)

    return df


def binary_presence(classes, active) -> tuple[list[str], list[bool]]:
    """Deduplicate objects per frame, marking each as active if seen active at least once."""
    objs: dict[str, int] = {}
    for c, a in zip(classes, active):
        objs[c] = objs.get(c, 0) + (1 if a else 0)
    return list(objs.keys()), [bool(x) for x in list(objs.values())]


def generate_binary_presence_df(
    df: pd.DataFrame, weighted: bool = False, weight: int | None = None
) -> pd.DataFrame:
    """Generates a dataframe with binary presence of objects per video."""

    df = df.copy()

    # get binary presence for each frame
    df["classes"], df["active"] = zip(
        *df.apply(lambda row: binary_presence(row["classes"], row["active"]), axis=1)
    )

    return generate_counts_df(df, weighted=weighted, weight=weight)


def generate_counts_df(
    df: pd.DataFrame, weighted: bool = False, weight: int | None = None
) -> pd.DataFrame:
    """Generates a dataframe with counts of active/inactive objects per video."""

    df = df.copy()

    def count_occurrences(classes, active):
        class_counts = Counter(classes)
        active_counts = Counter(
            {
                cls: sum([act and (cls == c) for act, c in zip(active, classes)])
                for cls in set(classes)
            }
        )
        return class_counts, active_counts

    df["class_counts"], df["active_counts"] = zip(
        *df.apply(lambda row: count_occurrences(row["classes"], row["active"]), axis=1)
    )

    # Create a new DataFrame from class_counts and active_counts
    if weighted:
        assert weight is not None, "weight must be provided when weighted=True"
        counts_df = pd.DataFrame(
            df.apply(
                lambda row: {
                    "adl": row["adl"],
                    "video": row["video"],
                    **{
                        f"count_{key}": value
                        + (weight * row["active_counts"].get(key, 0))
                        for key, value in row["class_counts"].items()
                    },
                },
                axis=1,
            ).tolist()
        )
    else:
        counts_df = pd.DataFrame(
            df.apply(
                lambda row: {
                    "adl": row["adl"],
                    "video": row["video"],
                    **{
                        f"count_{key}": value
                        for key, value in row["class_counts"].items()
                    },
                    **{
                        f"active_{key}": value
                        for key, value in row["active_counts"].items()
                    },
                },
                axis=1,
            ).tolist()
        )

    # Group by video and sum the values for each video
    grouped_counts_df = counts_df.groupby("video").agg(
        {
            **{"adl": "first"},
            **{col: "sum" for col in counts_df.columns if col not in ["adl", "video"]},
        }
    )

    return grouped_counts_df.reset_index()


def row_wise_min_max_scaling(df):
    """Perform row-wise min-max scaling on a dataframe."""
    # Extract the columns you want to scale (excluding non-numeric columns if any)
    columns_to_scale = df.select_dtypes(include="number").columns

    # Perform row-wise min-max scaling
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = df_scaled[columns_to_scale].apply(
        lambda row: (row - row.min()) / (row.max() - row.min()), axis=1
    )

    return df_scaled
