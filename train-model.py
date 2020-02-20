#! /bin/env python3
import pickle
import argparse
from tqdm import tqdm
from datetime import datetime

from happy_szczurki.models.cnn import ConvNet
from happy_szczurki.datasets import Dataset, DatasetIterator
from happy_szczurki.utils import smooth

from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

from sklearn.metrics import confusion_matrix, classification_report

LABELS_MAPPING = {
    None: 0,
    '22-kHz': 1,
    'SH': 2,
    'FM': 3,
    'RP': 4,
    'FL': 5,
    'ST': 6,
    'CMP': 7,
    'IU': 8,
    'TR': 9,
    'RM': 10
}


def load_pickled_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_new_model(args):
    net = NeuralNetClassifier(
        ConvNet,
        max_epochs=1,
        lr=0.001,
        train_split=CVSplit(3, stratified=False),
        optimizer=torch.optim.Adam
    )

    input_shape = (1, 65, 65)

    net.set_params(module__input_shape=input_shape, module__layer_params=[
        {
            "output_size": 32,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "max_pool_kernel": 3,
            "max_pool_stride": 2
        },
        {
            "output_size": 64,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "max_pool_kernel": 3,
            "max_pool_stride": 2
        },
        # {
        #     "output_size": 128,
        #     "kernel_size": 5,
        #     "stride": 1,
        #     "padding": 2,
        #     "max_pool_kernel": 3,
        #     "max_pool_stride": 2
        # },
    ], module__hidden_dim=[64, 64, 11])

    return net


def vizualize_history(model):
    colors = 'cmg'
    
    for c, metric in zip(colors, ('train_loss', 'valid_acc', 'valid_loss')):
        values = model.history[:, metric]
        window_size = min(len(values), 11)
        values_smooth = smooth(np.array(values), window_size, 'flat')[-len(values):]

        X = list(range(1, len(values)+1))
        plt.plot(X, values, f"{c}-", label=metric)
        plt.plot(X, values_smooth, f"{c}:", label=f"{metric}_smooth")
    
    plt.legend()
    plt.show()


def save_model(model, path=None):
        # save model
    if path is None:
        path = f"trained_models/{datetime.now()}.pkl"

    print(f"Saving model to: '{path}'")
    with open(path, "wb") as file:
        pickle.dump(model, file)   


def test_model(args):
    """Compute test scores of model on given dataset."""
    model = load_pickled_model(args.model)
    dataset = Dataset(args.dataset, use_mapping=LABELS_MAPPING)

    input_shape = model.get_params()['module__input_shape']

    # dataset = Dataset(args.test, use_mapping=LABELS_MAPPING)
    # print('normalization:', dataset.normalize(0.07912221, 0.5338538))

    # iterator = dataset.sample_iterator(args.samples, balanced=True, window_size=257, random_state=42)

    idx = np.array(range(0, len(dataset.y), 4))
    iterator = DatasetIterator(dataset, idx, batch_size=512, window_size=257)

    ys, y_preds = [], []

    with tqdm(total=len(idx)) as progress:
        for (X, y) in iterator:
            progress.update(X.shape[0])

            X = resize(X, (X.shape[0],) + input_shape[1:])                    
            X = X.reshape(-1, *input_shape)
            y = torch.from_numpy(y).long()
            
            y_pred = model.predict(X)

            ys.append(y)
            y_preds.append(y_pred)

    y = np.concatenate(ys)
    y_pred = np.concatenate(y_preds)

    print(X.shape, y.shape, y_pred.shape)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))


def train_model(args):
    """Train given model on dataset."""
    if args.model:
        model = load_pickled_model(args.pickle)
    else:
        model = build_new_model(args)

    dataset = Dataset(args.dataset, use_mapping=LABELS_MAPPING)
    input_shape = model.get_params()['module__input_shape']

    # print('Generating new data samples')
    
    iterator = dataset.sample_iterator(args.samples, window_size=257, batch_size=512, balanced=True, shuffle=True)

    try:
        for epoch in range(args.epochs):
            print(f"Starting epoch {epoch+1}/{args.epochs}")
            for (X, y) in iterator:
                X = resize(X, (X.shape[0],) + input_shape[1:])
                X = X.reshape(-1, *input_shape)
                y = torch.from_numpy(y).long()

                model.partial_fit(X, y)

                if model.history[-1,'valid_acc_best']:  # FIXME: use callbacks
                    save_model(model)

        vizualize_history(model)
    finally:
        save_model(model)


def inspect_model(args):
    model = load_pickled_model(args.model)
    vizualize_history(model)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--random-state', help='Initialize random state', )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # training
    parser_train = subparsers.add_parser('train', help='Train model instance')
    parser_train.add_argument('-p', '--model', metavar='PATH', help='Use already trained model from pickle file')
    parser_train.add_argument('-d', '--dataset', metavar='PATH', help='Train model on given dataset', type=str, required=True)
    parser_train.add_argument('-e', '--epochs', metavar='N', help='Number of training epoch', type=int, default=5)
    parser_train.add_argument('-s', '--samples', metavar='N', help='Number of samples generated for each epoch', type=int, default=10000)
    parser_train.add_argument('--balanced', metavar='BOOL', help='Perform balanced training (equalize class representants in each batch)', type=bool)

    parser_train.set_defaults(func=train_model)

    # Unsupported arguments
    # parser.add_argument('-b', '--base-class', help="Model base class")
    # parser.add_argument('-c', '--config', metavar='PATH', help='Path to JSON file with model parameters', type=str)

    # testing
    parser_test = subparsers.add_parser('test', help='Test performance of trained model instance')
    parser_test.add_argument('-m', '--model', metavar='PATH', help='Path to already trained model instance in pickle file', required=True)
    parser_test.add_argument('-d', '--dataset', metavar='PATH', help='Test model on given dataset', type=str, required=True)

    parser_test.set_defaults(func=test_model)

    # inspecting
    parser_inspect = subparsers.add_parser('inspect', help='Inspect properties of trained model instances')
    parser_inspect.add_argument('-m', '--model', metavar='PATH', help='Path to already trained model instance in pickle file', required=True)

    parser_inspect.set_defaults(func=inspect_model)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    args.func(args)