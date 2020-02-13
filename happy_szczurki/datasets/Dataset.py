from collections import Counter
from typing import Tuple

import librosa
import librosa.display
import numpy as np


class Dataset:
    def __init__(self, path):
        self.path = path

        dataset = dict(np.load(self.path, allow_pickle=True))

        self.X = dataset.pop('X')
        self.y = dataset.pop('y')
        self.meta = dataset.pop('meta')[()]

        assert len(dataset) == 0

        self.y_binary = np.where(self.y == None, 0, 1)

    def normalize(self, mean=None, std=None):
        # TODO: czy powinniśmy normalizować per współrzędna czy globalnie?
        if mean is None:
            mean = np.mean(self.X, axis=0)

        if std is None:
            std = np.std(self.X, axis=0, ddof=1)

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

        if with_idx:
            return idx, X, self.y[idx]

        return X, self.y[idx]

    def frame_to_time(self, frame: int) -> float:
        """Convert frame id to time in sec."""
        return librosa.core.frames_to_time(
            frame,
            sr=self.meta['sampling_rate'],
            hop_length=self.meta['hop_length'],
            n_fft=self.meta['n_fft']
        )
