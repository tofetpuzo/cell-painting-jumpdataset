"""
This script creates tiles from a composite image.
It takes a composite image and creates a specified number of tiles from it.
The tiles are saved in a specified directory.

Usage:
python create_tiles.py --composite_path <path_to_composite_image> --tile_dir <path_to_tile_directory> --num_tiles <number_of_tiles_to_generate>

Example:
python create_tiles.py --composite_path /path/to/composite_image.jpg --tile_dir /path/to/tile_directory --num_tiles 15
"""

import os
import numpy as np
import cv2
import pandas as pd
import logging


class CreateTiles:
    @staticmethod
    def create_tiles(composite: np.ndarray, tile_dir: str, num_tiles=16) -> int:
        """
        Create a specified number of tiles from a composite array.

        Args:
            composite (np.ndarray): The input composite array.
            tile_dir (str): The directory to save the tiles.
            num_tiles (int): Desired number of tiles to generate.
        """
        # Get composite dimensions
        height, width, _ = composite.shape
        os.makedirs(tile_dir, exist_ok=True)

        # Calculate grid dimensions for 1 tiles (3x5 grid)
        grid_rows = 4
        grid_cols = 4

        # Calculate tile dimensions
        tile_size_x = width // grid_cols
        tile_size_y = height // grid_rows

        # Ensure tiles are square
        tile_size = min(tile_size_x, tile_size_y)

        # Calculate strides to evenly space tiles
        stride_x = (
            (width - tile_size) // (grid_cols - 1) if grid_cols > 1 else tile_size
        )
        stride_y = (
            (height - tile_size) // (grid_rows - 1) if grid_rows > 1 else tile_size
        )

        tile_number = 0

        for i in range(0, height - tile_size + 1, stride_y):
            for j in range(0, width - tile_size + 1, stride_x):
                if tile_number >= num_tiles:
                    break

                # Extract the tile
                tile = composite[i : i + tile_size, j : j + tile_size, :]

                # Optionally resize tile to fixed size (128x128)
                tile = cv2.resize(tile, (128, 128), interpolation=cv2.INTER_CUBIC)

                # Save tile as parquet
                tile_filename = f"tile_{tile_number:04d}.parquet"
                tile_path = os.path.join(tile_dir, tile_filename)

                # Ensure we're using exactly 5 channels
                if tile.shape[-1] != 5:
                    logging.warning(
                        f"Expected 5 channels, got {tile.shape[-1]}. Adjusting..."
                    )
                    if tile.shape[-1] > 5:
                        tile = tile[..., :5]  # Take first 5 channels
                    else:
                        # Pad with zeros if less than 5 channels
                        padded = np.zeros((128, 128, 5), dtype=tile.dtype)
                        padded[..., : tile.shape[-1]] = tile
                        tile = padded

                # Reshape to 2D array (pixels x channels)
                reshaped_tile = tile.reshape(128 * 128, 5)
                reshaped_tile_df = pd.DataFrame(
                    reshaped_tile,
                    columns=[f"feature_{k}" for k in range(5)],
                )
                reshaped_tile_df.to_parquet(tile_path, compression="gzip")

                tile_number += 1

                # Break if we've reached the desired number of tiles
                if tile_number >= num_tiles:
                    break

        logging.info(f"Generated {tile_number} tiles in {tile_dir}.")

        return tile_number
