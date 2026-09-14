import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_embeddings(model, dataset, device, batch_size=256):
    """
    Extract penultimate-layer embeddings for all samples
    in a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN model.

    dataset : torch.utils.data.Dataset
        Dataset from which embeddings are extracted.

    device : torch.device
        CPU or GPU.

    batch_size : int
        Number of samples processed at one time.

    Returns
    -------
    embeddings : np.ndarray
        Embedding vector for every sample.

    labels : np.ndarray
        Ground-truth label for every sample.
    """

    model.eval()

    embeddings = []
    labels = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            if hasattr(model, "get_embedding"):
                features = model.get_embedding(x)
            else:
                features = model.extract_features(x)
            embeddings.append(
                features.cpu().numpy()
            )

            labels.append(
                y.numpy()
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return embeddings, labels
def extract_query_embedding(model, image, device):
    """
    Extract the embedding for one query image.
    """

    model.eval()

    image = image.to(device)

    # Add batch dimension
    image = image.unsqueeze(0)

    with torch.no_grad():
        embedding = model.get_embedding(image)

    return embedding.squeeze(0).cpu().numpy()