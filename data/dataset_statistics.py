from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_image_channel_statistics(images_dir):
    images_dir = Path(images_dir)
    channel_means = [0, 0, 0]
    channel_stds = [0, 0, 0]

    image_identifiers = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith(".")]
    image_count = len(image_identifiers)

    if not image_identifiers:
        raise RuntimeError(f"No input file found in {images_dir}, make sure you put your images there")

    for image_identifier in tqdm(image_identifiers):
        image_file = list(images_dir.glob(str(image_identifier) + ".*"))
        assert len(image_file) == 1, f"Either no image or multiple images found for the ID {image_identifier}: {image_file}"
        image_array = np.array(Image.open(image_file[0])).astype(np.float32) / 255.0
        for channel_index in range(3):
            channel_means[channel_index] += image_array[:, :, channel_index].mean()
            channel_stds[channel_index] += image_array[:, :, channel_index].std()

    channel_means = np.asarray(channel_means) / image_count
    channel_stds = np.asarray(channel_stds) / image_count

    print(f"normMean = {channel_means}")
    print(f"normStd = {channel_stds}")

    return channel_means, channel_stds
