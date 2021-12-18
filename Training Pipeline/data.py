import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm


def generate_noisy_images(scale_factor):
    raw_path = "./Data/Raw Images"
    label_path = "./Data/Labels"
    label_path_grayscale = "./Data/Labels Grayscale"
    image_path = "./Data/Images"
    image_path_grayscale = "./Data/Images Grayscale"
    raw_files = os.listdir("./Data/Raw Images")
    for file in tqdm(raw_files):
        img = cv2.imread(os.path.join(raw_path, file))
        old_size = img.shape[:2]
        ratio = float(512) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = 512 - new_size[1]
        delta_h = 512 - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h, w) = img.shape[:2]
        h -= int(h % scale_factor)
        w -= int(w % scale_factor)
        image = img[0:h, 0:w]
        image_grayscale = img_grayscale[0:h, 0:w]
        scaled_image = cv2.resize(
            image, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor, interpolation=cv2.INTER_CUBIC
        )
        scaled_image = cv2.resize(scaled_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        scaled_image_grayscale = cv2.resize(
            image_grayscale, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor, interpolation=cv2.INTER_CUBIC
        )
        scaled_image_grayscale = cv2.resize(
            scaled_image_grayscale, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(os.path.join(label_path, file), image)
        cv2.imwrite(os.path.join(label_path_grayscale, file), image_grayscale)
        cv2.imwrite(os.path.join(image_path, file), scaled_image)
        cv2.imwrite(os.path.join(image_path_grayscale, file), scaled_image_grayscale)


class DataLoader(Sequence):
    def __init__(self, file_ids, model_type, batch_size, shuffle=True):
        self.file_ids = file_ids
        self.model_type = model_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_ids)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        ids = [self.file_ids[i] for i in indexes]
        X, Y = self.__data_generation(ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        X, Y = [], []
        for _, id in enumerate(ids):
            image = cv2.imread(f"./Data/Images/{id}")
            label = cv2.imread(f"./Data/Labels/{id}")
            X.append(image)
            Y.append(label)
        return np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)
