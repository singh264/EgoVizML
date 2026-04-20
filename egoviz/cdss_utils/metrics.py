"""Module to compute metrics for the dashboard."""
import numpy as np


# percentage of frames with interaction
def interaction_percentage(frames, threshold=0.65):
    """Compute the percentage of frames with interaction."""

    # counters
    left_ints = 0
    right_ints = 0

    # iterate through frames
    for frame in frames:
        if frame["hands"] is not None:
            for hand in frame["hands"]:
                score = hand[4]
                interaction = hand[5] != "N"
                if score > threshold and interaction:
                    if hand[-1] == 0:  # left hand
                        left_ints += 1
                    elif hand[-1] == 1:  # right hand
                        right_ints += 1

    # compute percentage for each hand
    left_perc = left_ints / len(frames)
    right_perc = right_ints / len(frames)

    return left_perc, right_perc


def num_interactions(bool_list):
    consecutive_true_windows = []
    lengths = []

    window_start = None
    for i in range(len(bool_list)):
        if bool_list[i]:
            if window_start is None:
                window_start = i
        else:
            if window_start is not None:
                window_end = i - 1
                consecutive_true_windows.append((window_start, window_end))
                lengths.append(window_end - window_start + 1)
                window_start = None

    # Check if there is an unclosed window at the end of the list
    if window_start is not None:
        window_end = len(bool_list) - 1
        consecutive_true_windows.append((window_start, window_end))
        lengths.append(window_end - window_start + 1)

    num_windows = len(lengths)

    return num_windows, lengths


# number of interactions per hour by hand with frames sampled at 2 fps
def interactions_per_hour(frames, threshold=0.65, fps=2):
    """Compute the number of interactions per hour."""

    # counters
    left = []
    right = []

    # iterate through frames
    for frame in frames:
        if frame["hands"] is not None:
            for hand in frame["hands"]:
                score = hand[4] > threshold
                interaction = hand[5] != "N"
                if score and interaction:
                    if hand[-1] == 0:  # left hand
                        left.append(True)
                    elif hand[-1] == 1:  # right hand
                        right.append(True)
                else:
                    if hand[-1] == 0:
                        left.append(False)
                    elif hand[-1] == 1:
                        right.append(False)

    left_ints, _ = num_interactions(left)
    right_ints, _ = num_interactions(right)

    # compute number of interactions per hour
    left_ints_per_hour = left_ints / (len(frames) / fps) * 3600
    right_ints_per_hour = right_ints / (len(frames) / fps) * 3600

    return left_ints_per_hour, right_ints_per_hour


# average interaction duration by hand with frames sampled at 2 fps
# interaction duration is the number of consecutive frames in which the hand is interacting
def average_interaction_duration(frames, threshold=0.65, fps=2):
    """Compute the average interaction duration."""

    # counters
    left = []
    right = []

    # iterate through frames
    for frame in frames:
        if frame["hands"] is not None:
            for hand in frame["hands"]:
                score = hand[4] > threshold
                interaction = hand[5] != "N"
                if score and interaction:
                    if hand[-1] == 0:
                        left.append(True)
                    elif hand[-1] == 1:
                        right.append(True)
                else:
                    if hand[-1] == 0:
                        left.append(False)
                    elif hand[-1] == 1:
                        right.append(False)

    _, left_durs = num_interactions(left)
    _, right_durs = num_interactions(right)

    # compute average interaction duration
    left_avg_ints_duration = np.mean(left_durs) / fps
    right_avg_ints_duration = np.mean(right_durs) / fps

    return left_avg_ints_duration, right_avg_ints_duration
