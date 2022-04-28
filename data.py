import argparse
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

def create_parser():
    parser = argparse.ArgumentParser(description='Writes pre-processed data to disk.')
    parser.add_argument('--task',
                        choices=['flourescence'],
                        required=True,
                        help='Which task data to save')
    return parser

def write_df