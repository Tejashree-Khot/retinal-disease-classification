"""Download the IDRiD dataset from Kaggle."""

import kagglehub

path = kagglehub.dataset_download(
    "mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid"
)

print("Path to dataset files:", path)
