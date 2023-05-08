import os
import random
from skimage import io, img_as_ubyte
from skimage.color import rgba2rgb, rgb2lab, lab2rgb
from tqdm import tqdm
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt

DATA_DIR = "../data"
# DATA_DIR = "../../drive/Shareddrives"
TOTAL_LOADED = 15000  # Limit 99,990


def get_unique_ids(part):
    id_folders = os.listdir(DATA_DIR + "/raw/" + part)
    L_chans = []
    ab_chans = []

    for id in tqdm(id_folders, total=len(id_folders)):
        # Ignore hidden files
        if not id.startswith("."):
            all_faces = os.listdir(DATA_DIR + "/raw/" + part + "/" + id)
            # Randomly choose image for identity
            chosen_face = random.choice(all_faces)
            chosen_face = io.imread(DATA_DIR + "/raw/" + part + "/" +
                                    id + "/" + chosen_face)
            # Convert image from RGBA to Lab
            rgb_face = img_as_ubyte(rgba2rgb(chosen_face)).astype("float32")
            lab_face = rgb2lab(1.0 / 255 * rgb_face)

            L = lab_face[:, :, 0]
            L = np.expand_dims(L, axis=-1)
            ab = lab_face[:, :, 1:]
            ab /= 128

            L_chans.append(L)
            ab_chans.append(ab)

    print(f"Identities loaded: {len(L_chans)}")
    return np.array(L_chans), np.array(ab_chans)


def preprocess():
    parts = ["part1", "part2", "part3"]

    for i, part in enumerate(parts):
        print(f"Preprocessing part {i + 1}")
        L_chans, ab_chans = get_unique_ids(part)

        # Train/validation/test split: 80-10-10
        idxs = np.random.permutation(len(L_chans))
        train_i = math.floor(0.8 * len(L_chans))
        val_i = math.floor(0.9 * len(L_chans))

        train_idxs = idxs[:train_i]
        val_idxs = idxs[train_i:val_i]
        test_idxs = idxs[val_i:]

        # Store in file for train images
        train_L = L_chans[train_idxs]
        train_ab = ab_chans[train_idxs]
        print("Storing in file for train L channel")
        for L in tqdm(train_L, total=len(train_L)):
            store_data("normalized/train_L", L)
        print("Storing in file for train ab channel")
        for ab in tqdm(train_ab, total=len(train_ab)):
            store_data("normalized/train_ab", ab)

        # Store in file for validation images
        val_L = L_chans[val_idxs]
        val_ab = ab_chans[val_idxs]
        print("Storing in file for validation L channel")
        for L in tqdm(val_L, total=len(val_L)):
            store_data("normalized/val_L", L)
        print("Storing in file for validation ab channel")
        for ab in tqdm(val_ab, total=len(val_ab)):
            store_data("normalized/val_ab", ab)

        # Store in file for test images
        test_L = L_chans[test_idxs]
        test_ab = ab_chans[test_idxs]
        print("Storing in file for test L channel")
        for L in tqdm(test_L, total=len(test_L)):
            store_data("normalized/test_L", L)
        print("Storing in file for test ab channel")
        for ab in tqdm(test_ab, total=len(test_ab)):
            store_data("normalized/test_ab", ab)


def store_data(file_path, imgs):
    file = open(DATA_DIR + "/pickled/" + file_path, "ab")
    pickle.dump(imgs, file)
    file.close()


def load_data(file_path, num_imgs):
    imgs = []
    file = open(DATA_DIR + "/pickled/" + file_path, "rb")
    for _ in range(num_imgs):
        imgs.append([pickle.load(file)])  # Wrap for concatenation

    file.close()
    return np.concatenate(imgs)


# def get_LAB_images():
#     data = Datasets()
#     print("Storing training images")
#     for img in tqdm(data.train_color_imgs, total=len(data.train_color_imgs)):
#         lab_img = rgb2lab(img)
#         L = lab_img[:, :, 0]
#         L = np.expand_dims(L, axis=-1)
#         ab = lab_img[:, :, 1:]
#         store_data("train_L", [L])
#         store_data("train_ab", [ab])

#     print("Storing validation images")
#     for img in tqdm(data.val_color_imgs, total=len(data.val_color_imgs)):
#         lab_img = rgb2lab(img)
#         L = lab_img[:, :, 0]
#         L = np.expand_dims(L, axis=-1)
#         ab = lab_img[:, :, 1:]
#         store_data("val_L", [L])
#         store_data("val_ab", [ab])

#     print("Storing test images")
#     for img in tqdm(data.test_color_imgs, total=len(data.test_color_imgs)):
#         lab_img = rgb2lab(img)
#         L = lab_img[:, :, 0]
#         L = np.expand_dims(L, axis=-1)
#         ab = lab_img[:, :, 1:]
#         store_data("test_L", [L])
#         store_data("test_ab", [ab])


class Datasets():
    """ Class for containing the training, validation, and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self):
        train_n = math.floor(0.8 * TOTAL_LOADED)
        val_n = math.floor(0.9 * TOTAL_LOADED) - train_n
        test_n = TOTAL_LOADED - math.floor(0.9 * TOTAL_LOADED)
        # Load train images
        self.train_L = load_data("normalized/train_L", train_n)
        self.train_ab = load_data("normalized/train_ab", train_n)
        # Load validation images
        self.val_L = load_data("normalized/val_L", val_n)
        self.val_ab = load_data("normalized/val_ab", val_n)
        # Load test images
        self.test_L = load_data("normalized/test_L", test_n)
        self.test_ab = load_data("normalized/test_ab", test_n)


def main():
    # preprocess()
    data = Datasets()
    print(data.train_L.shape)


if __name__ == '__main__':
    main()
