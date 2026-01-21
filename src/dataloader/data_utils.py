"""Constants for class names and mapping used across the dataloader.

This module exposes two constants:

- CLASSES: ordered list of class names used by the dataset.
- CLASSES_DICT: mapping from class name (string) to integer label.

Keeping these definitions in a single module prevents hard-coded strings
scattered across the codebase and makes it easy to update label ordering.
"""

from collections import OrderedDict

CLASSES = [
    "Clinically_Significant_Macular_Edema",
    "No_DR",
    "normal",
    "Mild_Moderate_NPDR",
    "Severe_PDR",
]

CLASSES_DICT = OrderedDict(zip(CLASSES, range(len(CLASSES))))
