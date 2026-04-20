"""This module contains code for computing descriptive statistics from aim 1 quenstionnaires."""

import numpy as np

# dictionary of questions and their corresponding indices
questions = {
    0: "* I predominantly deal with technical systems because I am forced to.",
    1: "I enjoy spending time becoming acquainted with a new technical system.",
    2: "* It is enough for me that a technical system works; I do not care how or why.",
    3: "I try to understand how a technical system exactly works.",
    4: "* It is enough for me to know the basic functions of a technical system.",
    5: "I try to make full use of the capabilities of a technical system.",
    6: "I would personally use this dashboard regularly to understand and assess patient performance.",
    7: "I found the dashboard easy to use overall, and the various functions in this dashboard were well integrated.",
    8: "I felt like I could trust the metrics and information provided to me by the dashboard.",
    9: "* Using this dashboard regularly would substantially interrupt my workflow.",
    10: "* I feel apprehensive about using this dashboard with future patients.",
    11: "Using this dashboard is useful in understanding and assessing patient performance.",
    12: "* I found the dashboard very cumbersome to use, and the presented metrics were not clear.",
    13: "I would imagine that most people would learn to use this dashboard very quickly.",
    14: "The information presented by the dashboard would potentially influence patient therapy plans.",
    15: "* Presenting the information in this format is NOT useful to me or my assessment of patients.",
}

# list of questions to be reverse scored
reverse_scored = [0, 2, 4, 9, 10, 12, 15]


# scores for each question from each participant
# columns correspond to questions (above) and rows correspond to participants
scores = np.array(
    [
        [1, 4, 2, 4, 2, 5, 4, 4, 4, 2, 2, 5, 2, 4, 4, 1],  # T-01
        [1, 5, 1, 5, 1, 5, 5, 5, 5, 2, 1, 5, 1, 4, 5, 1],  # T-02
        [3, 2, 4, 2, 4, 3, 4, 4, 4, 3, 2, 4, 2, 4, 4, 2],  # T-03
        [3, 4, 2, 4, 2, 3, 5, 5, 4, 2, 2, 5, 2, 5, 5, 1],  # T-04
        [1, 2, 4, 2, 4, 2, 4, 4, 4, 2, 2, 4, 2, 3, 4, 2],  # T-05
    ]
)

# reverse score questions
for question in reverse_scored:
    scores[:, question] = 6 - scores[:, question]

# compute descriptive statistics
mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)
median_scores = np.median(scores, axis=0)
quantile_scores = np.quantile(scores, [0.25, 0.75], axis=0)
min_scores = np.min(scores, axis=0)
max_scores = np.max(scores, axis=0)


def get_descriptive_stats():
    """Return descriptive statistics for aim 1 questionnaires."""
    stats = {}
    for i, val in questions.items():
        stats[i] = {
            "question": val,
            "mean": mean_scores[i],
            "std": std_scores[i],
            "median": median_scores[i],
            "iqr": quantile_scores[1, i] - quantile_scores[0, i],
            "min": min_scores[i],
            "max": max_scores[i],
        }
    return stats
