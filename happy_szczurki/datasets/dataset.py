import warnings
from collections import Counter
from typing import Tuple, Optional, Dict

import librosa
import librosa.display
import numpy as np
from skimage.transform import resize



class Dataset:
    def __init__(self, path, use_mapping: Optional[Dict[Optional[str], int]] = None):
        self.path = path

        dataset = dict(np.load(self.path, allow_pickle=True))

        self.X = dataset.pop('X')
        self.y = dataset.pop('y')
        if use_mapping:
            self.y = np.array([use_mapping[label] for label in self.y])

        self.meta = dataset.pop('meta')[()]

        assert len(dataset) == 0

        self.y_binary = np.where(self.y == None, 0, 1)

    def normalize(self, mean=None, std=None):
        # TODO: czy powinniśmy normalizować per współrzędna czy globalnie?
        if mean is None:
            mean = np.mean(self.X)#, axis=0)

        if std is None:
            std = np.std(self.X)#, axis=0, ddof=1)

        # NOTE: inplace operators to prevent memory allocations
        self.X -= mean
        self.X /= std

        return mean, std


    def sample(self, n, *, balanced=False, with_idx=False, random_state=None, x_with_frame=False) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Choice `n` random samples from dataset.
        
        @param n: number of random samples to choose,
        @param balanced: if True number of samples for each class will be aprox. equal,
        @param with_idx: return data indexes of sampled records,
        @param random_state: random state used to generate samples indices,
        @param use_mapping: map label into ints
        """
        assert len(self.y.shape) == 1

        if balanced:
            counts = Counter(self.y)
            class_count = len(counts)

            # NOTE: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array/35216364
            probs = np.array([1.0 / (class_count * counts[x]) for x in self.y])
        else:
            probs = None

        idx = np.random.RandomState(random_state).choice(self.y.size, size=n, p=probs)

        new_X = np.pad(self.X, pad_width=[(128, 128), (0, 0)], mode='edge')

        if x_with_frame:
            X = np.array([new_X[index: index + 257] for index in idx])
        else:
            X = new_X[idx + 128]

        labels  = self.y[idx]

        if with_idx:
            return idx, X, labels

        return X, labels

    def sample_iterator(self, n, *, batch_size=512, balanced=False, window_size=1, random_state=None, **kwargs):
        if balanced:
            counts = Counter(self.y)
            class_count = len(counts)

            # NOTE: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array/35216364
            probs = np.array([1.0 / (class_count * counts[x]) for x in self.y])
        else:
            probs = None

        idx = np.random.RandomState(random_state).choice(self.y.size, size=n, p=probs)

        return DatasetIterator(self, idx, batch_size=batch_size, window_size=window_size, **kwargs)
    
    def frame_to_time(self, frame: int) -> float:
        """Convert frame id to time in sec."""
        return librosa.core.frames_to_time(
            frame,
            sr=self.meta['sampling_rate'],
            hop_length=self.meta['hop_length'],
            n_fft=self.meta['n_fft']
        )

class DatasetIterator:
    def __init__(self, dataset, indices, *, batch_size=512, window_size=1, shuffle=False, resize_to=None):
        self.dataset = dataset
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_to = resize_to

        self.left_pad = (window_size + 1 )//2
        self.window_size = window_size
        self.X_padded = np.pad(self.dataset.X, pad_width=[(self.left_pad, window_size - self.left_pad), (0, 0)], mode='edge')

        if not all(0 <= i < dataset.y.size for i in indices):
            raise IndexError()

        if len(indices) % batch_size != 0:
            warnings.warn("Last batch may be shortened!", RuntimeWarning)

    def __len__(self):
        return len(range(0, len(self.indices), self.batch_size))

    def __iter__(self):
        indices = np.array(self.indices)

        if self.shuffle:
            np.random.shuffle(indices)

        for batch_idx in range(0, len(indices), self.batch_size):
            indices_curr = indices[batch_idx: batch_idx + self.batch_size]

            X = np.array([self.X_padded[i: i + self.window_size] for i in indices_curr])
            if self.resize_to:
                X = resize(X, [X.shape[0]] + self.resize_to)
            y = self.dataset.y[indices_curr]
            
            yield X, y

