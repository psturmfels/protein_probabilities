import collections
import sys

import esm
import jax.numpy as jnp
import numpy as np
import torch
from torch.nn import Softmax

sys.path.append('/gscratch/cse/psturm/protein_probabilities/protein-embedding-retrieval/')

from contextual_lenses.encoders import one_hot_encoder


def compute_esm_embeddings(sequences,
                           max_token_count=10000,
                           rep_layer='logits'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    batches, sorted_indices = batch_sequence_data(sequences,
                                                  max_token_count=max_token_count)
    print(batches)
    esm_outputs = [None] * len(sequences)

    index_count = 0
    for data in batches:
        sequence_representations = batch_esm_outputs(data, model, batch_converter,
                                                     rep_layer=rep_layer, device=device)
        for seq_rep in sequence_representations:
            esm_outputs[sorted_indices[index_count]] = seq_rep
            index_count += 1

    return esm_outputs


def batch_sequence_data(sequences,
                        max_token_count=10000):
    sequence_lengths = [len(seq) for seq in sequences]
    sorted_indices = np.argsort(sequence_lengths)

    batches = []
    current_batch = []

    for index in sorted_indices:
        seq_length = sequence_lengths[index]

        if (len(current_batch) + 1) * seq_length > max_token_count:
            batches.append(current_batch)
            current_batch = []

        current_batch.append((f'seq{index}', sequences[index]))

    batches.append(current_batch)
    return batches, sorted_indices


def batch_esm_outputs(data, model, batch_converter, rep_layer='logits', device='cpu'):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    tokenized_lengths = torch.sum(batch_tokens > 2, dim=-1).detach().numpy()

    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    if rep_layer == 'logits':
        softmax_fn = Softmax(dim=-1)
        outputs = results['logits']
        outputs = softmax_fn(outputs)
    else:
        outputs = results["representations"][rep_layer]

    sequence_representations = []
    for i, seq_len in enumerate(tokenized_lengths):
        sequence_representations.append(outputs[i, 1: seq_len + 1].cpu().detach().numpy())

    return sequence_representations


def flatten(x, padding_mask=None):
    """Apply padding and flatten over sequence length axis."""

    if padding_mask is not None:
        x = x * padding_mask

    rep = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    return rep


def flattened_one_hot_encoder(batch_inds, num_categories):
    """Flattens padded one-hot encoding from jax.nn."""

    padding_mask = jnp.expand_dims(jnp.where(batch_inds < num_categories - 1, 1, 0), axis=2)

    one_hots = one_hot_encoder(batch_inds, num_categories)
    flattened_one_hots = flatten(one_hots, padding_mask)

    return flattened_one_hots


def get_num_params(model):
    """Computes number of parameters in flax model."""

    # Code source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    def dict_flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(dict_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    params = model.params
    params = dict_flatten(params)

    num_params = 0
    for layer in params.keys():
        num_params += np.prod(params[layer].shape)

    return num_params


if __name__ == '__main__':
    sequence = 'KALTARQQEVFDLIRDKALTARQQEVFDLIRD'
    sequences = [sequence[:i] for i in range(len(sequence))]

    outputs = compute_esm_embeddings(sequences, max_token_count=len(sequence))
    print(sequences)
    print([o.shape for o in outputs])
