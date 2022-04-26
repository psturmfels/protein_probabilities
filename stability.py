import pandas as pd
from tape.datasets import LMDBDataset
from protein_lm import domains

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


def gfp_dataset_to_df(in_name):
    dataset = LMDBDataset(in_name)
    df = pd.DataFrame(list(dataset)[:])
    df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
    return df


def main():


if __name__ == '__main__':
    main()
