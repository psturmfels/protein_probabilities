'''
Code imported from
https://github.com/googleinterns/protein-embedding-retrieval/blob/master/cnn_protein_landscapes.ipynb
'''
import sys
sys.path.append('/gscratch/cse/psturm/protein_probabilities/protein-embedding-retrieval/')

import pandas as pd
import argparse

import matplotlib.pyplot as plt

import scipy.stats

import sklearn
from sklearn.linear_model import Ridge
import pprint

from contextual_lenses.loss_fns import mse_loss
from contextual_lenses.train_utils import create_data_iterator, \
    create_optimizer, create_representation_model, train
from contextual_lenses.encoders import cnn_one_hot_encoder
from contextual_lenses.contextual_lenses import max_pool, linear_max_pool
from protein_lm import domains
from tape.datasets import LMDBDataset

from utils import *

GFP_SEQ_LEN = 237
GFP_AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', '-'
]
GFP_PROTEIN_DOMAIN = domains.VariableLengthDiscreteDomain(
    vocab=domains.ProteinVocab(include_anomalous_amino_acids=False,
                               include_eos=True,
                               include_pad=True),
    length=GFP_SEQ_LEN)


def create_parser():
    parser = argparse.ArgumentParser(description='Trains a model on the Rocklin Flourescence data.')
    parser.add_argument('--model',
                        choices=['cnn_maxpool', 'regression'],
                        required=True,
                        help='Which model to train')
    return parser

def gfp_seq_to_inds(seq):
    """Encode GFP amino acid sequence."""

    return GFP_PROTEIN_DOMAIN.encode([seq])[0]


def gfp_dataset_to_df(in_name):
    dataset = LMDBDataset(in_name)
    df = pd.DataFrame(list(dataset)[:])
    df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
    return df


def create_gfp_df(test=False):
    """Processes GFP data into a featurized dataframe."""

    if test:
        gfp_df = gfp_dataset_to_df('data/fluorescence/fluorescence_test.lmdb')
    else:
        gfp_df = gfp_dataset_to_df('data/fluorescence/fluorescence_train.lmdb')

    gfp_df['one_hot_inds'] = gfp_df.primary.apply(lambda x: gfp_seq_to_inds(x[:GFP_SEQ_LEN]))

    return gfp_df


def create_gfp_batches(batch_size, epochs=1, test=False, buffer_size=None,
                       seed=0, drop_remainder=False):
    """Creates iterable object of GFP batches."""

    if test:
        buffer_size = 1

    gfp_df = create_gfp_df(test=test)

    fluorescences = gfp_df['log_fluorescence'].values

    gfp_batches = create_data_iterator(df=gfp_df, input_col='one_hot_inds',
                                       output_col='log_fluorescence',
                                       batch_size=batch_size, epochs=epochs,
                                       buffer_size=buffer_size, seed=seed,
                                       drop_remainder=drop_remainder)

    return gfp_batches, fluorescences


def gfp_evaluate(predict_fn, title, batch_size=256,
                 test_data=None, pred_fluorescences=None,
                 clip_min=-999999, clip_max=999999, show_figs=False):
    """Computes predicted fluorescences and measures performance in MSE and spearman correlation."""

    test_batches, test_fluorescences = create_gfp_batches(batch_size=batch_size,
                                                          test=True, buffer_size=1)

    if test_data is not None:
        test_batches = test_data

    if pred_fluorescences is None:
        pred_fluorescences = []
        for batch in iter(test_batches):
            X, Y = batch
            preds = predict_fn(X)
            for pred in preds:
                pred_fluorescences.append(pred[0])

    pred_fluorescences = np.array(pred_fluorescences)
    pred_fluorescences = np.clip(pred_fluorescences, clip_min, clip_max)

    spearmanr = scipy.stats.spearmanr(test_fluorescences, pred_fluorescences).correlation
    mse = sklearn.metrics.mean_squared_error(test_fluorescences, pred_fluorescences)

    bright_inds = np.where(test_fluorescences > 2.5)
    bright_test_fluorescences = test_fluorescences[bright_inds]
    bright_pred_fluorescences = pred_fluorescences[bright_inds]
    bright_spearmanr = scipy.stats.spearmanr(bright_test_fluorescences, bright_pred_fluorescences).correlation
    bright_mse = sklearn.metrics.mean_squared_error(bright_test_fluorescences, bright_pred_fluorescences)

    dark_inds = np.where(test_fluorescences < 2.5)
    dark_test_fluorescences = test_fluorescences[dark_inds]
    dark_pred_fluorescences = pred_fluorescences[dark_inds]
    dark_spearmanr = scipy.stats.spearmanr(dark_test_fluorescences, dark_pred_fluorescences).correlation
    dark_mse = sklearn.metrics.mean_squared_error(dark_test_fluorescences, dark_pred_fluorescences)

    results = {
        'title': title,
        'spearmanr': round(spearmanr, 3),
        'mse': round(mse, 3),
        'bright_spearmanr': round(bright_spearmanr, 3),
        'bright_mse': round(bright_mse, 3),
        'dark_spearmanr': round(dark_spearmanr, 3),
        'dark_mse': round(dark_mse, 3),
    }

    pprint.pprint(results)

    if show_figs:
        plt.scatter(test_fluorescences, pred_fluorescences, s=1, alpha=0.5)
        plt.xlabel('True LogFluorescence')
        plt.ylabel('Predicted LogFluorescence')
        plt.title(title)
        plt.show()

        plt.scatter(bright_test_fluorescences, bright_pred_fluorescences, s=1, alpha=0.5)
        plt.xlabel('True LogFluorescence')
        plt.ylabel('Predicted LogFluorescence')
        bright_title = title + ' (Bright)'
        plt.title(bright_title)
        plt.show()

        plt.scatter(dark_test_fluorescences, dark_pred_fluorescences, s=1, alpha=0.5)
        plt.xlabel('True LogFluorescence')
        plt.ylabel('Predicted LogFluorescence')
        dark_title = title + ' (Dark)'
        plt.title(dark_title)
        plt.show()

    return results, pred_fluorescences


def linear_regression():
    gfp_train_df = create_gfp_df()
    gfp_test_df = create_gfp_df(test=True)

    train_fluorescences = gfp_train_df['log_fluorescence']
    gfp_train_one_hot_inds = np.array([x for x in gfp_train_df['one_hot_inds'].values])
    gfp_train_one_hots = flattened_one_hot_encoder(gfp_train_one_hot_inds,
                                                   num_categories=len(GFP_AMINO_ACID_VOCABULARY))
    gfp_test_one_hot_inds = np.array([x for x in gfp_test_df['one_hot_inds'].values])
    gfp_test_one_hots = flattened_one_hot_encoder(gfp_test_one_hot_inds, num_categories=len(GFP_AMINO_ACID_VOCABULARY))

    gfp_linear_model = Ridge()
    gfp_linear_model.fit(X=gfp_train_one_hots, y=train_fluorescences)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    linear_model_pred_fluorescences = gfp_linear_model.predict(gfp_test_one_hots)
    gfp_linear_model_results, linear_model_pred_fluorescences = \
        gfp_evaluate(predict_fn=None,
                     title='Linear Regression',
                     pred_fluorescences=linear_model_pred_fluorescences,
                     clip_min=min(train_fluorescences),
                     clip_max=max(train_fluorescences))
    print('Number of Parameters for Fluorescence Linear Regression: ' + str(len(gfp_linear_model.coef_)))


def cnn_maxpool():
    epochs = 50
    gfp_train_batches, train_fluorescences = create_gfp_batches(batch_size=256, epochs=epochs)

    layers = ['CNN_0', 'Dense_1']
    learning_rate = [1e-3, 5e-6]
    weight_decay = [0.0, 0.05]

    encoder_fn = cnn_one_hot_encoder
    encoder_fn_kwargs = {
        'n_layers': 1,
        'n_features': [1024],
        'n_kernel_sizes': [5],
        'n_kernel_dilations': None
    }
    reduce_fn = max_pool
    reduce_fn_kwargs = {

    }
    loss_fn_kwargs = {

    }

    gfp_model = create_representation_model(encoder_fn=encoder_fn,
                                            encoder_fn_kwargs=encoder_fn_kwargs,
                                            reduce_fn=reduce_fn,
                                            reduce_fn_kwargs=reduce_fn_kwargs,
                                            num_categories=len(GFP_AMINO_ACID_VOCABULARY),
                                            output_features=1)

    gfp_optimizer = train(model=gfp_model,
                          train_data=gfp_train_batches,
                          loss_fn=mse_loss,
                          loss_fn_kwargs=loss_fn_kwargs,
                          learning_rate=learning_rate,
                          weight_decay=weight_decay,
                          layers=layers)

    gfp_results, pred_fluorescences = gfp_evaluate(predict_fn=gfp_optimizer.target,
                                                   title='CNN + MaxPool',
                                                   batch_size=256,
                                                   clip_min=min(train_fluorescences),
                                                   clip_max=max(train_fluorescences))
    print('Number of Parameters for Fluorescence CNN + MaxPool: ' + str(get_num_params(gfp_optimizer.target)))

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.model == 'regression':
        linear_regression()
    elif args.model == 'cnn_maxpool':
        cnn_maxpool()
    else:
        raise ValueError(f'Unrecognized model `{args.model}`')


if __name__ == '__main__':
    main()
