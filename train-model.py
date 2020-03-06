#! /bin/env python3
"""
TODO:
"""
# Hide sklearn warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import pickle
import random
import os
import json
import re
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

from happy_szczurki.datasets import Dataset, DatasetIterator, LABELS_MAPPING, REV_LABELS_MAPPING
from happy_szczurki.models.cnn import ConvNet
from happy_szczurki.utils import smooth, Table


def load_pickled_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_new_model(args, config, with_tensorboard=False):
    # TODO: use checkpoints
    # cp = Checkpoint(f_pickle='')
    # train_end_cp = TrainEndCheckpoint(f_pickle='')
    if with_tensorboard:
        writer = SummaryWriter(f'runs/{datetime.now()}')
        tensorboard = TensorBoard(writer)

    module_name, class_name = config.pop('module').rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_name)

    net = NeuralNetClassifier(
        getattr(module, class_name),
        max_epochs=1,
        lr=0.001,
        train_split=CVSplit(4, stratified=True),
        optimizer=torch.optim.Adam,
        # optimizer__weight_decay=0.001,
        criterion=torch.nn.CrossEntropyLoss,
        # callbacks=[tensorboard],
    )

    input_shape = tuple(config['module__input_shape'])
    
    net.set_params(**config)

    # NOTE: https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d
    # NOTE: https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2
    #   suggest that dropout on CNN is wrong idea

    # TODO: smote 
    # TODO: globalne konwolucje
    # TODO: statystyki kroczące w BatchNorm, można dodać BatchNorm do liniowych warstw
    # TODO: sprawdzić czy filtr 5x5 pomaga, czy stridy pomagają
    # TODO: EarlyStopping
    # TODO: Learning Rate Scheduler / cosinus

    # TODO: jak sobie radzic z niezbalansowanymi danymi
    #  - data augmentation
    #  - parametr `weight` do CrossEntropyLoss
    
    # TODO: Modele:
    #  - CNN
    #  - RCNN
    #  - dwógłowy CNN
    #  - LSTM 
    #  - SVM
    #  - SVM z PCA
    #  - prosty model od dr. Dudy (znajdowanie maksa w kolumnie)
    # TODO: spotkanie dr. Spurek - 12,14
    # TODO: Dekonwolucje
    # QUESTION: co trzeba zrobic do EEMLa

    if with_tensorboard:
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

    for c, metric in zip('ry', ('l1-norm', 'l2-norm')):
        values = model.history[:, metric]
        if values:
            plt.plot(X, values / np.max(values), label=f"scaled {metric}")

    plt.legend()
    plt.show()


def save_model(model, path=None):
        # save model
    if path is None:
        path = f"trained_models/{datetime.now()}.pkl"

    print(f"Saving model to: '{path}'")
    with open(path, "wb") as file:
        pickle.dump(model, file)   


def create_classification_report(y_true, y_pred, format='text') -> str:
    metrics = ['precision', 'recall', 'f1-score', 'support']
    labels = list(REV_LABELS_MAPPING.keys())

    table = Table([''] + metrics, title='Classification report')
    for column in table.columns:
        column.align = 'right'

    for key, values in classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True).items():
        if key.isdigit():
            key = REV_LABELS_MAPPING[int(key)]

        if key == 'micro avg':
            table.insert_separator()

        table << [key] + [values[metric] for metric in metrics]
    
    return str(table)


def create_confusion_matrix(y_true, y_pred) -> str:
    labels = list(REV_LABELS_MAPPING.keys())

    table = Table(['true 𝈏 predicted'] + [REV_LABELS_MAPPING[key] for key in labels], title='Confusion matrix')
    for column in table.columns:
        column.align = 'right'
        column.min_width = 3

    for label, row in zip(labels, confusion_matrix(y_true, y_pred, labels=labels)):
        table << ([REV_LABELS_MAPPING[label]] + list(row))
    
    return str(table)


def test_model(args):
    """Compute test scores of model on given dataset."""
    model = load_pickled_model(args.model)
    model_name = os.path.splitext(args.model)
    report_path = model_name[0] + '_' + str(datetime.now()) + '.txt'

    with open(report_path, 'w') as output:
        for dataset_path in args.dataset:
            dataset_name = os.path.basename(dataset_path)

            dataset = Dataset(dataset_path, use_mapping=LABELS_MAPPING)

            input_shape = model.get_params()['module__input_shape']

            # iterator = dataset.sample_iterator(args.samples, balanced=True, window_size=257, random_state=42)

            idx = np.array(range(0, len(dataset.y), 5))
            iterator = DatasetIterator(dataset, idx, batch_size=512, window_size=257, resize_to=input_shape[1:])

            ys, y_preds = [], []

            with tqdm(total=len(idx), leave=False, desc=dataset_name) as progress:
                for (X, y) in iterator:
                    progress.update(X.shape[0])

                    X = X.reshape(-1, *input_shape)
                    y = torch.from_numpy(y).long()
                    
                    y_pred = model.predict(X)

                    ys.append(y)
                    y_preds.append(y_pred)

            y_true = np.concatenate(ys)
            y_pred = np.concatenate(y_preds)

            # Generate test report
            report = "\n".join([
                '\n',
                f" Testing on {dataset_name} ".center(100, '-'),
                '',
                create_classification_report(y_true, y_pred),
                create_confusion_matrix(y_true, y_pred),
            ])
            output.write(report)
            print(report, end="")


def computer_model_norm(model, norm=2):
    norm_val = None
    for param in model.parameters():
        if norm_val is None:
            norm_val = param.norm(norm)
        else:
            norm_val += param.norm(norm)
    return norm_val.detach().numpy()


def train_model(args):
    """Train given model on dataset."""
    if args.model:
        model = load_pickled_model(args.model)
    else:
        with open(args.config, 'r') as input_:
            content = input_.read()
            # HACK: Handle non-RFC compilant comments
            content = re.sub(r'//.*', '', content)
            config = json.loads(content)
            model = build_new_model(args, config)

    inspect_model_layers(model)

    datasets = [Dataset(path, use_mapping=LABELS_MAPPING) for path in args.dataset]
    input_shape = model.get_params()['module__input_shape']

    output_path = f"trained_models/{datetime.now()}_%s.pkl"
    
    try:
        for epoch in range(args.epochs):
            dataset = random.choice(datasets)
            iterator = dataset.sample_iterator(args.samples, window_size=257, batch_size=args.batch_size, balanced=args.balanced, shuffle=True, resize_to=input_shape[1:])

            print(f"Starting epoch {epoch+1}/{args.epochs}")
            for (X, y) in iterator:
                X = X.reshape(-1, *input_shape)
                y = torch.from_numpy(y).long()

                model.partial_fit(X, y)

                model.history.record('l1-norm', computer_model_norm(model.module_, 1.0))
                l2 = computer_model_norm(model.module_, 2.0)
                model.history.record('l2-norm', l2)
                print(l2)

                if model.history[-1,'valid_loss_best']:  # FIXME: use callbacks
                    save_model(model, output_path % 'best')

        vizualize_history(model)
    finally:
        save_model(model, output_path % 'final')


def vizualize_filters(model):
    for layer_no, layer in enumerate(model.module_.layers):
        if 'Conv2d' in str(type(layer)):
            weight = layer.weight

            _fig, axs = plt.subplots(8, 8)
            axs = [axs[i,j] for i in range(8) for j in range(8)]
            idx = 0
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    data = weight[i,j].detach().numpy()
                    if idx < len(axs):
                        axs[idx].imshow(data, cmap='gray')
                    idx += 1
            plt.suptitle(f"Layer #{layer_no}: {layer}")
            plt.show()

def inspect_model_layers(model):
    t = Table(['#', 'layer', 'params', 'output_shape'])

    from happy_szczurki.utils import calculate_output_sizes
    input_shape = model.get_params()['module__input_shape']
    output_sizes = calculate_output_sizes(input_shape, model.module_.layers)

    total_params = 0
    for i, (layer, shape) in enumerate(zip(model.module_.layers, output_sizes)):
        total_params += (params := sum(param.nelement() for param in layer.parameters()))
        t << (i, layer, params, shape)

    t.insert_separator()
    t << ('', 'Total parameters count:', total_params, '')
    print(t)


def inspect_model(args):
    model = load_pickled_model(args.model)
    # show model architecture
    # print(model.module_.layers)
    
    params = model.get_params()
    t = Table(['key', 'value'])
    import pprint
    for key, value in params.items():
        if key in ('optimizer_', 'criterion_',) or key.startswith('module__'):
            # print(key, value)
            t << (key, pprint.pformat(value))
    print(t)

    inspect_model_layers(model)
       
    # params_count = sum([param.nelement() for param in model.module_.parameters()])
    # print(f"Number of parameters: {params_count}")

    vizualize_history(model)
    vizualize_filters(model)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--random-state', help='Initialize random state', )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # training
    parser_train = subparsers.add_parser('train', help='Train model instance')
    group = parser_train.add_mutually_exclusive_group(required=True)

    group.add_argument('-m', '--model', metavar='PATH', help='Use already trained model from pickle file')
    group.add_argument('-c', '--config', metavar='PATH', help='Path to JSON file with model parameters', type=str)

    parser_train.add_argument('-d', '--dataset', metavar='PATH', help='Train model on given datasets', nargs='+', type=str, required=True)
    parser_train.add_argument('-e', '--epochs', metavar='N', help='Number of training epoch', type=int, default=5)
    parser_train.add_argument('-s', '--samples', metavar='N', help='Number of samples generated for each epoch', type=int, default=10000)
    parser_train.add_argument('-b', '--batch-size', metavar='N', help='Number of samples in each batch', type=int, default=512)
    parser_train.add_argument('--balanced', metavar='BOOL', help='Perform balanced training (equalize class representants in each batch)', type=bool, default=True)

    parser_train.set_defaults(func=train_model)

    # Unsupported arguments
    # parser.add_argument('-b', '--base-class', help="Model base class")

    # testing
    parser_test = subparsers.add_parser('test', help='Test performance of trained model instance')
    parser_test.add_argument('-m', '--model', metavar='PATH', help='Path to already trained model instance in pickle file', required=True)
    parser_test.add_argument('-d', '--dataset', metavar='PATH', help='Test model on given dataset', type=str, nargs='+', required=True)
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
    print(args, end="\n\n")

    args.func(args)
