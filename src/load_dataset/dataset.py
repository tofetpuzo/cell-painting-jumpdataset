import os
import torchvision.transforms as transforms
import pyarrow.parquet as pq
import torch
import random


# Dataset class
class TiledDataset(torch.utils.data.Dataset):
    def __init__(self, file_dir, augment=True):

        # get all the files in the directory some are nested
        all_files = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                all_files.append(os.path.relpath(os.path.join(root, file), file_dir))

        self.tile_files = []

        # Filter valid parquet files
        for f in all_files:
            file_path = os.path.join(file_dir, f)
            if os.path.getsize(file_path) > 0:
                try:
                    # Try to read the parquet file header to validate it
                    pq.read_metadata(file_path)
                    self.tile_files.append(f)
                except Exception as e:
                    print(f"Skipping invalid parquet file {f}: {str(e)}")

        print(
            f"Total files: {len(all_files)}, Valid parquet files: {len(self.tile_files)}"
        )
        self.file_dir = file_dir
        self.augment = augment
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomVerticalFlip(p=0.3),
                ]
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        file = self.tile_files[idx]
        file_path = os.path.join(self.file_dir, file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")

        try:
            table = pq.read_table(file_path)
            load_df_parquet_tiles = table.to_pandas()
            restored_tiled_array = load_df_parquet_tiles.to_numpy()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            # Skip this file and get another one
            return self.__getitem__((idx + 1) % len(self))
        tile = restored_tiled_array.reshape(128, 128, 5)
        tile = torch.from_numpy(tile).permute(2, 0, 1).float()

        if self.transform:
            tile = self.transform(tile)
            tile = self.rotate_90_180_270(tile)

        return tile, file_path

    def rotate_90_180_270(self, image, probability=0.3):
        if random.random() < probability:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            rotation = transforms.RandomRotation(degrees=(angle, angle))
            return rotation(image)
        return image
