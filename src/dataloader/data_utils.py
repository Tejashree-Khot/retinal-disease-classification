"""Constants for class names and mapping used across the dataloader.

This module exposes two constants:

- CLASSES: ordered list of class names used by the dataset.
- CLASSES_DICT: mapping from class name (string) to integer label.

Keeping these definitions in a single module prevents hard-coded strings
scattered across the codebase and makes it easy to update label ordering.
"""

from collections import OrderedDict

RETINOPATHY_GRADES = ["0", "1", "2", "3", "4"]
RISK_OF_MACULAR_EDEMA = ["0", "1", "2"]
DIAGNOSTIC_CLASSES = ["No_DR", "Mild_Moderate_NPDR", "Clinically_Significant_Macular_Edema", "Severe_PDR"]

ANNOTATIONS_COLUMN_INDEX_MAP = {
    "Retinopathy grade": RETINOPATHY_GRADES,
    "Risk of macular edema": RISK_OF_MACULAR_EDEMA,
    "class": DIAGNOSTIC_CLASSES,
}

# Change column name to change the label column
LABEL_COLUMN_NAME = "class"

CLASSES = ANNOTATIONS_COLUMN_INDEX_MAP[LABEL_COLUMN_NAME]
CLASSES_DICT = OrderedDict(zip(CLASSES, range(len(CLASSES))))

ZERO_SHOT_CLASS_PROMPTS = [
    "a retinal fundus photograph with no signs of diabetic retinopathy",
    "a retinal fundus photograph showing mild to moderate non-proliferative diabetic retinopathy",
    "a retinal fundus photograph showing clinically significant macular edema",
    "a retinal fundus photograph showing severe proliferative diabetic retinopathy",
]
