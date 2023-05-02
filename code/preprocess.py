import os
import random
from skimage import io, img_as_float32
from skimage.color import rgba2rgb, rgb2gray, rgb2lab
from tqdm import tqdm
import numpy as np
import pickle
import math

DEFAULT_DATA_DIR = "../data"
TOTAL_LOADED = 10000  # Limit 99,990


def get_unique_ids(part, data_dir=DEFAULT_DATA_DIR):
    id_folders = os.listdir(data_dir + "/raw/" + part)
    rgb_imgs = []
    gray_imgs = []

    for id in tqdm(id_folders, total=len(id_folders)):
        # Ignore hidden files
        if not id.startswith("."):
            all_faces = os.listdir(data_dir + "/raw/" + part + "/" + id)
            # Randomly choose image for identity
            chosen_face = random.choice(all_faces)
            chosen_face = io.imread(data_dir + "/raw/" + part + "/" +
                                    id + "/" + chosen_face)
            # Convert image from RGBA to RGB and grayscale
            rgb_face = rgba2rgb(chosen_face)
            rgb_face = img_as_float32(rgb_face)
            gray_face = rgb2gray(rgb_face)
            gray_face = img_as_float32(gray_face)
            gray_face = np.expand_dims(gray_face, axis=-1)

            rgb_imgs.append(rgb_face)
            gray_imgs.append(gray_face)

    print(f"Identities loaded: {len(rgb_imgs)}")
    return np.array(rgb_imgs), np.array(gray_imgs)


def preprocess():
    parts = ["part1", "part2", "part3"]

    for i, part in enumerate(parts):
        print(f"Preprocessing part {i + 1}")
        rgb_imgs, gray_imgs = get_unique_ids(part)

        # Train/validation/test split: 80-10-10
        idxs = np.random.permutation(len(rgb_imgs))
        train_i = math.floor(0.8 * len(rgb_imgs))
        val_i = math.floor(0.9 * len(rgb_imgs))

        train_idxs = idxs[:train_i]
        val_idxs = idxs[train_i:val_i]
        test_idxs = idxs[val_i:]

        # Store in file for train images
        train_rgb_imgs = rgb_imgs[train_idxs]
        train_gray_imgs = gray_imgs[train_idxs]
        print("Storing in file for train RGB images")
        store_data("train_rgb", train_rgb_imgs)
        print("Storing in file for train gray images")
        store_data("train_gray", train_gray_imgs)
        # Store in file for validation images
        val_rgb_imgs = rgb_imgs[val_idxs]
        val_gray_imgs = gray_imgs[val_idxs]
        print("Storing in file for validation RGB images")
        store_data("validation_rgb", val_rgb_imgs)
        print("Storing in file for validation gray images")
        store_data("validation_gray", val_gray_imgs)
        # Store in file for test images
        test_rgb_imgs = rgb_imgs[test_idxs]
        test_gray_imgs = gray_imgs[test_idxs]
        print("Storing in file for test RGB images")
        store_data("test_rgb", test_rgb_imgs)
        print("Storing in file for test gray images")
        store_data("test_gray", test_gray_imgs)


def store_data(file_path, imgs, data_dir=DEFAULT_DATA_DIR):
    file = open(data_dir + "/pickled/" + file_path, "ab")
    pickle.dump(imgs, file)
    file.close()


def load_data(data_dir, filename, num_imgs):
    imgs = []
    file = open(data_dir + "/pickled/" + filename, "rb")
    for _ in range(num_imgs):
        imgs.append(pickle.load(file))

    file.close()
    return np.concatenate(imgs)


def get_LAB_images():
    data = Datasets()
    print("Storing training images")
    for img in tqdm(data.train_color_imgs, total=len(data.train_color_imgs)):
        lab_img = rgb2lab(img)
        L = lab_img[:, :, 0]
        L = np.expand_dims(L, axis=-1)
        ab = lab_img[:, :, 1:]
        store_data("train_L", [L])
        store_data("train_ab", [ab])

    print("Storing validation images")
    for img in tqdm(data.val_color_imgs, total=len(data.val_color_imgs)):
        lab_img = rgb2lab(img)
        L = lab_img[:, :, 0]
        L = np.expand_dims(L, axis=-1)
        ab = lab_img[:, :, 1:]
        store_data("val_L", [L])
        store_data("val_ab", [ab])

    print("Storing test images")
    for img in tqdm(data.test_color_imgs, total=len(data.test_color_imgs)):
        lab_img = rgb2lab(img)
        L = lab_img[:, :, 0]
        L = np.expand_dims(L, axis=-1)
        ab = lab_img[:, :, 1:]
        store_data("test_L", [L])
        store_data("test_ab", [ab])


class Datasets():
    """ Class for containing the training, validation, and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_dir):
        train_n = math.floor(0.8 * TOTAL_LOADED)
        val_n = math.floor(0.9 * TOTAL_LOADED) - train_n
        test_n = TOTAL_LOADED - math.floor(0.9 * TOTAL_LOADED)
        # Load train images
        self.train_L = load_data(data_dir, "train_L", train_n)
        self.train_ab = load_data(data_dir, "train_ab", train_n)
        # Load validation images
        self.val_L = load_data(data_dir, "val_L", val_n)
        self.val_ab = load_data(data_dir, "val_ab", val_n)
        # Load test images
        self.test_L = load_data(data_dir, "test_L", test_n)
        self.test_ab = load_data(data_dir, "test_ab", test_n)


def main():
    data = Datasets()
    print(f"Train L: {data.train_L.shape}")
    print(f"Train ab: {data.train_ab.shape}")
    print(f"Val L: {data.val_L.shape}")
    print(f"Val ab: {data.val_ab.shape}")
    print(f"Test L: {data.test_L.shape}")
    print(f"Test ab: {data.test_ab.shape}")


if __name__ == '__main__':
    main()
