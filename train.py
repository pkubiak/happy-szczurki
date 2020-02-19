import pickle
import argparse
from datetime import datetime

from happy_szczurki.models.CNN import ConvNet
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

# TODO: tesotwanie
"""
test_dataset = Dataset('data/converted/ch1-2018-11-20_10-29-02_0000012.wav.npz')

X_all_test, y_all_test = test_dataset.sample(test_dataset.X.shape[0] // 100, balanced=False, x_with_frame=True, use_mapping=mapping)
for i in range((X_all_test.shape[0] // N) - 1):
    i = i * N
    X, y = X_all_test[i:i + N], y_all_test[i:i + N]
    X = resize(X, (N,) + input_shape[1:])
    X = X.reshape(-1, *input_shape)
    y = torch.from_numpy(y).long()
    
    print(net.score(X, y))
"""
# TODO: plotowanie lossów


def load_trained_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def build_new_model(args):
    net = NeuralNetClassifier(
        ConvNet,
        max_epochs=5,
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--partial', help="Use partially trained model")
    parser.add_argument('-e', '--epochs', help='Number of training epoch', type=int, default=10)
    parser.add_argument('-s', '--samples', help='Number of samples generated for each epoch', type=int, default=500)
    parser.add_argument('-t', '--test', metavar='PATH', help='Test model on given dataset', type=str)
    return parser.parse_args()

def save_model(model, path=None):
        # save model
    if path is None:
        path = f"trained_models/{datetime.now()}.pkl"

    print(f"Saving model to: '{path}'")
    with open(path, "wb") as file:
        pickle.dump(model, file)   


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.partial:
        model = load_trained_model(args.partial)
    else:
        model = build_new_model(args)

    if args.test:
        vizualize_history(model)

        input_shape = model.get_params()['module__input_shape']

        dataset = Dataset(args.test, use_mapping=LABELS_MAPPING)
        dataset.normalize()

        iterator = dataset.sample_iterator(args.samples, balanced=True, window_size=257, random_state=42)

        idx = np.array(range(0, len(dataset.y), 4))
        # print(len(idx), len(dataset.y))
        iterator = DatasetIterator(dataset, idx, batch_size=512, window_size=257)

        ys, y_preds = [], []

        for (X, y) in iterator:
            # print('--', X.shape)
            X = resize(X, (X.shape[0],) + input_shape[1:])
            # for i in range(len(y)):
            #     if y[i] == 2:
            #         # print('2')
            #         plt.imshow(X[i])
            #         plt.show()
                    
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
        

    else:        
        # print(model.history[:, 'train_loss'])
        dataset = Dataset('data/converted/ch1-2018-11-20_10-20-34_0000006.wav.npz', use_mapping=LABELS_MAPPING)
        print('normalization:', dataset.normalize())

        input_shape = model.get_params()['module__input_shape']

        print('Generating new data samples')

        
        iterator = dataset.sample_iterator(args.samples, window_size=257, batch_size=512, balanced=True, shuffle=True)

        try:

            for epoch in range(args.epochs):
                print(f"Starting epoch {epoch+1}/{args.epochs}")
                for (X, y) in iterator:
                    print(X.shape)
                    X = resize(X, (X.shape[0],) + input_shape[1:])
                    # print(X.shape)
                    X = X.reshape(-1, *input_shape)
                    y = torch.from_numpy(y).long()

                    model.partial_fit(X, y)

                    if model.history[-1,'valid_acc_best']:
                        save_model(model)
            vizualize_history(model)
        finally:
            save_model(model)