#! /bin/env python3
import argparse
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, TensorBoard, TrainEndCheckpoint
from skorch.dataset import CVSplit
from tqdm import tqdm

from happy_szczurki.datasets import Dataset, DatasetIterator, LABELS_MAPPING
from happy_szczurki.models.cnn import ConvNet
from happy_szczurki.utils import smooth


def load_pickled_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_new_model(args):
    # TODO: use checkpoints
    # cp = Checkpoint(f_pickle='')
    # train_end_cp = TrainEndCheckpoint(f_pickle='')
    writer = SummaryWriter(f'runs/{datetime.now()}')
    tensorboard = TensorBoard(writer)

    net = NeuralNetClassifier(
        ConvNet,
        max_epochs=1,
        lr=0.001,
        train_split=CVSplit(3, stratified=True),
        optimizer=torch.optim.Adam,
        callbacks=[tensorboard],
    )

    # TODO: How do I apply L2 regularization?
    input_shape = (1, 65, 65)

    net.set_params(
        module__input_shape=input_shape,
        module__classes=11,
        module__conv_layers=[
            # NOTE: https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2
            #   suggest that dropout on CNN is wrong idea
            dict(
                conv=dict(out_channels=32, kernel_size=5, stride=1, padding=2),
                activation='relu',
                pool=dict(type='max', kernel_size=3, stride=2),
                batch_norm=True,
                dropout=0.0,
            ),
            dict(
                conv=dict(out_channels=32, kernel_size=5, stride=1, padding=2),
                activation='relu',
                pool=dict(type='max', kernel_size=3, stride=2),
                batch_norm=True,
                dropout=0.0,
            ),
        ],
        module__linear_layers=[
            dict(out_features=64, activation='relu', dropout=0.2),
            dict(out_features=64, activation='relu', dropout=0.2),
        ]
    )

    # print(net.module_)
    # breakpoint()
    writer.add_graph(net.module_, torch.zeros(size=(1, ) + input_shape))

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
    iterator = DatasetIterator(dataset, idx, batch_size=512, window_size=257, resize_to=input_shape[1:])

    ys, y_preds = [], []

    with tqdm(total=len(idx)) as progress:
        for (X, y) in iterator:
            progress.update(X.shape[0])

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
        model = load_pickled_model(args.model)
    else:
        model = build_new_model(args)

    dataset = Dataset(args.dataset, use_mapping=LABELS_MAPPING)
    input_shape = model.get_params()['module__input_shape']

    output_path = f"trained_models/{datetime.now()}_%s.pkl"
    

    try:
        for epoch in range(args.epochs):
            iterator = dataset.sample_iterator(args.samples, window_size=257, batch_size=args.batch_size, balanced=args.balanced, shuffle=True, resize_to=input_shape[1:])
            print(f"Starting epoch {epoch+1}/{args.epochs}")
            for (X, y) in iterator:
                X = X.reshape(-1, *input_shape)
                y = torch.from_numpy(y).long()

                model.partial_fit(X, y)

                if model.history[-1,'valid_loss_best']:  # FIXME: use callbacks
                    save_model(model, output_path % 'best')

        vizualize_history(model)
    finally:
        save_model(model, output_path % 'final')


def vizualize_filters(model):
    for layer in model.module_.layers:
        if 'Conv2d' in str(type(layer)):
            weight = layer.weight
            print(layer.weight.shape)

            _fig, axs = plt.subplots(8, 8)
            axs = [axs[i,j] for i in range(8) for j in range(8)]
            idx = 0
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    data = weight[i,j].detach().numpy()
                    if idx < len(axs):
                        axs[idx].imshow(data, cmap='gray')
                    idx += 1
            plt.suptitle(f"{layer}")
            plt.show()


def inspect_model(args):
    model = load_pickled_model(args.model)
    # show model architecture
    print(model.module_.layers)
    params_count = sum([param.nelement() for param in model.module_.parameters()])
    print(f"Number of parameters: {params_count}")

    vizualize_history(model)
    vizualize_filters(model)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--random-state', help='Initialize random state', )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # training
    parser_train = subparsers.add_parser('train', help='Train model instance')
    parser_train.add_argument('-m', '--model', metavar='PATH', help='Use already trained model from pickle file')
    parser_train.add_argument('-d', '--dataset', metavar='PATH', help='Train model on given dataset', type=str, required=True)
    parser_train.add_argument('-e', '--epochs', metavar='N', help='Number of training epoch', type=int, default=5)
    parser_train.add_argument('-s', '--samples', metavar='N', help='Number of samples generated for each epoch', type=int, default=10000)
    parser_train.add_argument('-b', '--batch-size', metavar='N', help='Number of samples in each batch', type=int, default=512)
    parser_train.add_argument('--balanced', metavar='BOOL', help='Perform balanced training (equalize class representants in each batch)', type=bool, default=True)

    parser_train.set_defaults(func=train_model)

    # Unsupported arguments
    # parser.add_argument('-b', '--base-class', help="Model base class")
    # parser.add_argument('-c', '--config', metavar='PATH', help='Path to JSON file with model parameters', type=str)

    # testing
    parser_test = subparsers.add_parser('test', help='Test performance of trained model instance')
    parser_test.add_argument('-m', '--model', metavar='PATH', help='Path to already trained model instance in pickle file', required=True)
    parser_test.add_argument('-d', '--dataset', metavar='PATH', help='Test model on given dataset', type=str, required=True)
    # TODO: save output to file 

    parser_test.set_defaults(func=test_model)

    # inspecting
    parser_inspect = subparsers.add_parser('inspect', help='Inspect properties of trained model instances')
    parser_inspect.add_argument('-m', '--model', metavar='PATH', help='Path to already trained model instance in pickle file', required=True)

    parser_inspect.set_defaults(func=inspect_model)

    # TODO: annotate


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    args.func(args)
