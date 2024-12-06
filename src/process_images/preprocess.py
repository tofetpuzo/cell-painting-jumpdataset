"""Create numpy arrays from TIFF files"""
import logging
import re
from skimage.transform import resize
import os
import numpy as np
from PIL import Image
import gzip

from create_tiles import CreateTiles


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log_error.log"), logging.StreamHandler()],
)


file_mappings = [
    {
        "filename": "r01c02f01p01-ch5sk1fk1fl1.tiff",
        "regex": r"r\d{2}c\d{2}f\d{2}p\d{2}-ch(\d+)sk\d+fk\d+fl\d+\.tiff",
        "match_channel": r"-ch(\d+)",
        "prefix": r"r\d{2}c\d{2}f\d{2}p\d{2}",
        "OrigER": 1,
        "OrigRNA": 2,
        "OrigDNA": 4,
        "OrigAGP": 5,
        "OrigMito": 3,
        "source_folder": 1,
    },
]

# Utility functions


class TiffProcessor:
    def __init__(self, directory: str):
        """
        Initialize the TiffProcessor with a directory.

        Parameters:
        directory (str): The directory where the TIFF files are located.
        """
        self.directory = directory

    @staticmethod
    def load_channels(group_matching_files_in_five: list, mapping: dict) -> np.ndarray:
        """
        Load different channels from TIFF files into separate arrays.
        Args:
          group_matching_files_in_five (list): List of matching files
          mapping (dict): Mapping of channel numbers to channel names
        Returns:
        np.ndarray: A 6-channel array containing the loaded channels.
        """
        channel_images = {}
        try:
            for file_in_group in group_matching_files_in_five:
                channel_number = re.search(
                    mapping.get("match_channel"), file_in_group)
                if channel_number is None:
                    continue  # Skip
                channel_number = int(channel_number.group(1))
                if (
                    channel_number
                    in {
                        mapping["OrigER"],
                        mapping["OrigRNA"],
                        mapping["OrigDNA"],
                        mapping["OrigAGP"],
                        mapping["OrigMito"],
                    }
                    and channel_number not in channel_images
                ):
                    img_array = np.array(Image.open(file_in_group))
                    img_array = TiffProcessor.normalize_tiff(img_array)
                    channel_images[channel_number] = img_array

            return TiffProcessor.create_6_channel_array(channel_images)
        except Exception as e:
            logging.error(f"Error in load_channels: {e}")

    @staticmethod
    def normalize_tiff(img_array: np.ndarray) -> np.ndarray:
        """
        Normalize and resize a TIFF image array.

        Parameters:
        img_array (np.ndarray): The image array to normalize and resize.

        Returns:
        np.ndarray: The normalized and resized image array.
        """
        try:
            # Find the minimum and maximum pixel values in the image
            min_val = np.min(img_array)
            max_val = np.max(img_array)

            if min_val == max_val:
                # If min and max are the same, the image is constant (all pixels have the same value)
                # In this case, normalization would cause division by zero, so we return a zero array instead
                normalized = np.zeros_like(img_array, dtype=np.uint8)
            else:
                # Normalize the image to the range [0, 1]
                # This is done by subtracting the minimum value and then dividing by the range
                normalized = (img_array - min_val) / (max_val - min_val)

                # Clip values to ensure they're in the range [0, 1]
                # This handles any potential floating point errors
                normalized = np.clip(normalized, 0, 1)

                # Scale the normalized values to the range [0, 255] and convert to 8-bit unsigned integer
                normalized = (normalized * 255).astype(np.uint8)

            # Resize the image to (1080, 1080) if it's not already that size
            if normalized.shape[:2] != (1080, 1080):
                normalized = resize(normalized, (1080, 1080),
                                    preserve_range=True).astype(np.uint8)

            return normalized
        except Exception as e:
            logging.error(f"Error in normalize_tiff: {e}")
            return None

    @staticmethod
    def create_6_channel_array(channel_images: dict) -> np.ndarray:
        """
        Create a 6-channel array from the provided channel images.

        Parameters:
        channel_images (dict): A dictionary where keys are channel numbers and values are image arrays.

        Returns:
        np.ndarray: A 6-channel composite image array.
        """
        # Assume the image dimensions are 1080x1080
        try:
            num_channels = 6
            image_shape = (1080, 1080)
            # 1280 × 1080

            # Initialize a zero array for the composite image with 6 channels
            composite_image = np.zeros(
                (*image_shape, num_channels), dtype=np.uint8)

            # Map input channels to output channels in desired order: DNA, ER, RNA, AGP, Mito
            channel_mapping = {
                5: 0,  # DNA -> channel 1
                4: 1,  # ER -> channel 2
                6: 2,  # RNA -> channel 3
                2: 3,  # AGP -> channel 4
                1: 4,  # Mito -> channel 5
            }

            # Populate the array with provided channels in the desired order
            for input_channel, output_idx in channel_mapping.items():
                composite_image[..., output_idx] = channel_images.get(
                    input_channel, np.zeros(image_shape, dtype=np.uint8))
            return composite_image
        except Exception as e:
            logging.error(f"Error in create_6_channel_array: {e}")


def main():
    # Directory containing TIFF files
    source_directory = "C:\plot_neural\cell-painting-jumpdataset\sample_images"
    tile_output_dir = "tiles_output"  # Directory to save tiles
    tile_size = 128
    overlap = 0.5

    # Instantiate the TiffProcessor
    processor = TiffProcessor(directory=source_directory)

    # Traverse the directory and process TIFF files
    for mapping in file_mappings:
        # Gather all TIFF files matching the regex in the source directory
        all_files = [
            os.path.join(source_directory, f)
            for f in os.listdir(source_directory)
            if re.search(mapping["regex"], f)
        ]

        # Group files by prefix (assuming prefix is unique per group)
        grouped_files = {}
        for file in all_files:
            prefix = re.search(mapping["prefix"], file).group()
            if prefix not in grouped_files:
                grouped_files[prefix] = []
            grouped_files[prefix].append(file)

        # Process each group of matching files
        for prefix, group_files in grouped_files.items():
            logging.info(f"Processing group: {prefix}")
            try:
                # Create a composite array
                composite_array = processor.load_channels(group_files, mapping)

                if composite_array is None:
                    logging.warning(
                        f"Failed to create composite for group {prefix}")
                    continue

                # Create tiles from the composite
                output_dir = os.path.join(tile_output_dir, prefix)
                CreateTiles.create_tiles(
                    composite=composite_array,
                    tile_dir=output_dir,
                )
            except Exception as e:
                logging.error(f"Error processing group {prefix}: {e}")


if __name__ == "__main__":
    main()
