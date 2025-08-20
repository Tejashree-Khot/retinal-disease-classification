import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid"
)

print("Path to dataset files:", path)
