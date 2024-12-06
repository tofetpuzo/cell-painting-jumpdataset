import os
import numpy as np
import cv2
import pandas as pd
import logging


class CreateTiles:
    @staticmethod
    def create_tiles(composite: np.ndarray, tile_dir, num_tiles=15):
        """
        Create a specified number of tiles from a composite array.

        Args:
            composite (np.ndarray): The input composite array.
            tile_dir (str): The directory to save the tiles.
            num_tiles (int): Desired number of tiles to generate.
        """
        # Get composite dimensions
        height, width, channels = composite.shape
        os.makedirs(tile_dir, exist_ok=True)

        # Calculate approximate tile dimensions to generate the desired number of tiles
        tile_size_x = max(1, width // int(np.sqrt(num_tiles)))
        tile_size_y = max(1, height // int(np.sqrt(num_tiles)))

        # Ensure tiles are square (optional)
        tile_size = min(tile_size_x, tile_size_y)

        # Calculate stride to space tiles evenly
        stride_x = (width - tile_size) // (int(np.sqrt(num_tiles)
                                               ) - 1) if num_tiles > 1 else tile_size
        stride_y = (height - tile_size) // (int(np.sqrt(num_tiles)
                                                ) - 1) if num_tiles > 1 else tile_size

        tile_number = 0

        for i in range(0, height - tile_size + 1, stride_y):
            for j in range(0, width - tile_size + 1, stride_x):
                if tile_number >= num_tiles:
                    break

                # Extract the tile
                tile = composite[i:i + tile_size, j:j + tile_size, :]

                # Optionally resize tile to fixed size (e.g., 128x128)
                tile = cv2.resize(tile, (128, 128),
                                  interpolation=cv2.INTER_CUBIC)

                # Save tile as parquet
                tile_filename = f"tile_{tile_number:04d}.parquet"
                tile_path = os.path.join(tile_dir, tile_filename)

                reshaped_tile = tile.reshape(-1, tile.shape[-1])
                reshaped_tile_df = pd.DataFrame(
                    reshaped_tile, columns=[
                        f"feature_{k}" for k in range(tile.shape[-1])]
                )
                reshaped_tile_df.to_parquet(tile_path, compression="gzip")

                tile_number += 1

                # Break if we've reached the desired number of tiles
                if tile_number >= num_tiles:
                    break

        logging.info(f"Generated {tile_number} tiles in {tile_dir}.")


# def load_compressed_npy(filename):
#     with gzip.GzipFile(filename, "r") as f:
#         return np.load(f)


# class CreateTiles:
#     @staticmethod
#     def create_tiles(directory, tile_dir, tile_size=128, overlap=0):
#         stride = int(tile_size * (1 - overlap))
#         files = glob.glob(os.path.join(directory, "*.npy.gz"))
#         for file_path in files:
#             # Extract the base name of the file
#             try:
#                 base_name = os.path.splitext(
#                     os.path.splitext(os.path.basename(file_path))[0])[0]
#                 # Load the data array
#                 data = load_compressed_npy(file_path)

#                 # Resize each channel
#                 resized_array = np.zeros((512, 512, 5))
#                 for i in range(5):
#                     # Resize one channel at a time
#                     resized_array[:, :, i] = cv2.resize(
#                         data[:, :, i], (512, 512), interpolation=cv2.INTER_CUBIC)

#                 # Calculate number of tiles in each dimension
#                 num_tiles_x = (
#                     resized_array.shape[0] - tile_size) // stride + 1
#                 num_tiles_y = (
#                     resized_array.shape[1] - tile_size) // stride + 1

#                 tile_number = 0
#                 temp_tile_dir = os.path.join(tile_dir, base_name)
#                 os.makedirs(temp_tile_dir, exist_ok=True)

#                 # Loop over each possible tile position
#                 for i in range(num_tiles_x):
#                     for j in range(num_tiles_y):
#                         tile = resized_array[
#                             i * stride: i * stride + tile_size,
#                             j * stride: j * stride + tile_size,
#                             :,
#                         ]

#                         tile_filename = f"{base_name}_{tile_number:04}"
#                         tile_path = os.path.join(temp_tile_dir, tile_filename)

#                         # Save the tile to disk temporarily
#                         # np.save(tile_path, cp.asnumpy(tile))
#                         # Shape becomes (16384, 5)
#                         reshaped_tile = tile.reshape(-1, tile.shape[-1])
#                         reshaped_tile_df = pd.DataFrame(
#                             reshaped_tile, columns=[f"feature_{i}" for i in range(tile.shape[-1])])

#                         # Save the DataFrame as a compressed Parquet file
#                         reshaped_tile_df.to_parquet(
#                             f"{tile_path}.parquet", compression="gzip")

#                         tile_number += 1

#             except Exception as e:
#                 print(f"Error with {file_path}, {e}")


# # Example usage
# if __name__ == "__main__":
#     CreateTiles.create_tiles(
#         directory="/home/jovyan/Cell-painting/Cell-Painting/cell-painting/src/train/source_1/organized/",
#         tile_dir="/home/jovyan/Cell-painting/Cell-Painting/cell-painting/src/train/source_1/organized/",
#         tile_size=128,
#         overlap=0,
#     )
