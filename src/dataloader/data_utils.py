"""Constants for class names and mapping used across the dataloader.

This module exposes two constants:

- CLASSES: ordered list of class names used by the dataset.
- CLASSES_DICT: mapping from class name (string) to integer label.

Keeping these definitions in a single module prevents hard-coded strings
scattered across the codebase and makes it easy to update label ordering.
"""

from collections import OrderedDict

CLASSES = ["No_DR", "Mild_Moderate_NPDR", "Clinically_Significant_Macular_Edema", "Severe_PDR"]
# CLASSES = ["0", "1", "2"]
CLASSES = ["0", "1", "2"]

CLASSES_DICT = OrderedDict(zip(CLASSES, range(len(CLASSES))))
