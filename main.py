"""
This script preprocesses TIFF files and creates tiles from them.
It loads different channels from TIFF files, normalizes and resizes them, and creates tiles from the composite image.
The resulting tiles are saved in a specified directory."""

from src.process_images.create_tiles import CreateTiles
import argparse
import os
import re
import logging

from src.process_images.preprocess import TiffProcessor
from src.generate.generate import generate_embeddings


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


def main():
    # Directory containing TIFF files
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        required=True,
        help="Path to the directory containing images",
    )
    args = parser.parse_args()

    source_directory = args.directory
    tile_output_dir = "tiles_output"  # Directory to save tiles

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
                    logging.warning(f"Failed to create composite for group {prefix}")
                    continue

                # Create tiles from the composite
                output_dir = os.path.join(tile_output_dir, prefix)
                CreateTiles.create_tiles(
                    composite=composite_array,
                    tile_dir=output_dir,
                )
            except Exception as e:
                logging.error(f"Error processing group {prefix}: {e}")

    # using the generated tiles, generate embeddings

    generate_embeddings(
        source_folder=tile_output_dir,
        batch_size=64,
        return_reconstruction=False,
        return_reconstruction_error=False,
        device=None,
    )


if __name__ == "__main__":
    main()
