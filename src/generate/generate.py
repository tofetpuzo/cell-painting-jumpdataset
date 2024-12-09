# tissue_vae/generate.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.generate.weights.model import VAE
from src.load_dataset.dataset import TiledDataset
import os


def generate_embeddings(
    source_folder,
    batch_size=64,
    return_reconstruction=False,
    return_reconstruction_error=False,
    device=None,
):
    """
    Generate embeddings for processed image tiles.

    Args:
        source_folder (str): Path to folder containing processed image tiles
        batch_size (int): Batch size for processing
        return_reconstruction (bool): Whether to return reconstructed images
        return_reconstruction_error (bool): Whether to return reconstruction error
        device (str): Device to use for computation ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing requested outputs:
            - 'embeddings': numpy array of shape (n_samples, 512)
            - 'reconstructions': numpy array of shape (n_samples, 128, 128, 5) if requested
            - 'reconstruction_error': numpy array of shape (n_samples,) if requested
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = VAE().to(device)

    # find the full path traverse the directory
    model_path = Path(__file__).parent / "weights" / "model_weights.pth"

    if not model_path.exists():
        raise FileNotFoundError("Model weights not found. Please download them first.")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Setup dataset and dataloader
    dataset = TiledDataset(source_folder, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    embeddings = []
    reconstructions = [] if return_reconstruction else None
    reconstruction_errors = [] if return_reconstruction_error else None

    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            recon_batch, mean, _ = model(batch)

            embeddings.append(mean.cpu().numpy())

            if return_reconstruction:
                reconstructions.append(recon_batch.cpu().numpy())

            if return_reconstruction_error:
                error = torch.mean((recon_batch - batch) ** 2, dim=(1, 2, 3))
                reconstruction_errors.append(error.cpu().numpy())

    # Combine results
    results = {"embeddings": np.concatenate(embeddings, axis=0)}

    if return_reconstruction:
        results["reconstructions"] = np.concatenate(reconstructions, axis=0)

    if return_reconstruction_error:
        results["reconstruction_error"] = np.concatenate(reconstruction_errors, axis=0)

    # write the embeddings to a file

    embeddings_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "embeddings.npy"
    )
    np.save(embeddings_path, results["embeddings"])

    return results
