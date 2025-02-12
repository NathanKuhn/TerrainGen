from PIL import Image, ImageFilter
import numpy as np
import albumentations as A
import os
from multiprocessing import Pool

RAW_DATA_PATH = "raw/"
DATASET_PATH = "dataset/"
SAMPLE_COUNT = 100  # Number of samples to take from each image
NUM_WORKERS = 8

TRANSFORM = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
    ]
)


def get_image_paths():
    return [
        RAW_DATA_PATH + filename
        for filename in os.listdir(RAW_DATA_PATH)
        if filename.endswith(".tif")
    ]


def load_all_images():
    images = []
    for filename in os.listdir(RAW_DATA_PATH):
        if filename.endswith(".tif"):
            image = Image.open(RAW_DATA_PATH + filename)
            images.append(np.array(image))

    return np.array(images)


def save_dataset(images):
    np.save(DATASET_PATH + "terrain_dataset.npy", images)


def process_image(args):
    i, image_path = args
    image = np.array(Image.open(image_path))

    samples = []
    for _ in range(SAMPLE_COUNT):
        sample = TRANSFORM(image=image)["image"]

        if np.std(sample) > 50:
            samples.append(sample)

    print(f"Image: {i:3} | Samples: {len(samples):3}")
    return np.array(samples)


def main():
    image_paths = get_image_paths()
    print(
        f"Gathering {SAMPLE_COUNT} samples each for {len(image_paths)} images with {NUM_WORKERS} workers"
    )

    with Pool(NUM_WORKERS) as p:
        samples = p.map(process_image, enumerate(image_paths))

    samples = np.concatenate([s for s in samples if len(s) > 0])

    print(f"Saving dataset with {samples.shape[0]} total samples")
    save_dataset(samples)


if __name__ == "__main__":
    main()
