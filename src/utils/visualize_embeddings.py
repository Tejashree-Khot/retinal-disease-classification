import os

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.simple_model import SimpleCNN


def visualize_cnn_embeddings(
    data_dir="../data/test",
    model: SimpleCNN | None = None,
    model_path: str | None = None,
    max_images: int = 20,
    batch_size: int = 32,
    image_size: int = 224,
    device: str = "cpu",
    perplexity: float = 5.0,
    random_state: int = 42,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Extract convolutional embeddings from a SimpleCNN and visualize them with t-SNE.

    Returns (embeddings_2d, labels) as numpy arrays.
    """
    if model is None:
        model = SimpleCNN()
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    dataset = ImageFolder(os.path.join(data_dir), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embeddings = []
    labels = []
    with torch.no_grad():
        collected = 0
        for images, lbls in dataloader:
            images = images.to(device)
            feats = model.conv(images)  # Get feature maps before linear layer
            feats = feats.view(feats.size(0), -1)  # Flatten
            take = min(images.size(0), max_images - collected)
            embeddings.append(feats[:take])
            labels.append(lbls[:take])
            collected += take
            for i in range(take):
                print(
                    f"Processed image {collected - take + i + 1} with shape: {images[i].shape}, label: {lbls[i]}"
                )
            if collected >= max_images:
                break

    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Embeddings and labels concatenated and converted to numpy.")

    # Adjust perplexity if needed (must be < n_samples)
    effective_perplexity = min(perplexity, max(1.0, len(embeddings) - 1))

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE transformation complete.")

    if show or save_path:
        plt.figure(figsize=(8, 6))
        for label in sorted(set(labels)):
            idx = labels == label
            plt.scatter(
                embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f"Class {label}", alpha=0.6
            )
        plt.legend()
        plt.title("t-SNE Visualization of CNN Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print("Plot displayed and/or saved.")

    return embeddings_2d, labels


# Optional CLI usage
if __name__ == "__main__":
    # Example call replicating former defaults
    visualize_cnn_embeddings()
