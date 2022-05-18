import argparse
import os
from collections import Counter
from typing import Iterable

import numpy as np
from Bio import SeqIO
from fairseq.data import Dictionary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Downloads data for PLM pre-training or fine-tuning')
    parser.add_argument('--dataset',
                        choices=['example', 'uniref50'],
                        default='example',
                        help='What dataset to download (usually fasta files)')
    parser.add_argument('--download_dir',
                        type=str,
                        default='/gscratch/cse/psturm/data/proteins',
                        help='Where to download the data to (top level directory).')
    return parser


def get_download_commands(dataset: str, download_path: str) -> str:
    if dataset == 'example':
        download_target = os.path.join(download_path, 'example.fasta')
        if os.path.exists(download_target):
            raise FileExistsError(f'File {download_target} already exists!')

        return f'''
            wget https://raw.githubusercontent.com/soedinglab/MMseqs2/master/examples/DB.fasta &&
            mv DB.fasta {download_target}
        '''
    elif dataset == 'uniref50':
        download_target = os.path.join(download_path, 'uniref50.fasta')
        if os.path.exists(download_target):
            raise FileExistsError(f'File {download_target} already exists!')

        return f'''
                    wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz &&
                    gzip -d file.gz uniref50.fasta.gz &&
                    mv uniref50.fasta {download_target}
                '''
    else:
        raise ValueError(f'Unrecognized dataset `{dataset}`')


def split_data(dataset: str,
               download_path: str,
               valid_percentage: float = 0.05):
    if dataset == 'example':
        download_target = os.path.join(download_path, 'example.fasta')
    elif dataset == 'uniref50':
        download_target = os.path.join(download_path, 'uniref50.fasta')
    else:
        raise ValueError(f'Unrecognized dataset `{dataset}`')

    train_target = os.path.join(download_path, 'train.fasta')
    valid_target = os.path.join(download_path, 'valid.fasta')

    all_records = list(SeqIO.parse(download_target, 'fasta'))
    np.random.shuffle(all_records)
    num_valid_records = int(len(all_records) * valid_percentage)
    valid_records = all_records[:num_valid_records]
    train_records = all_records[num_valid_records:]

    SeqIO.write(train_records, train_target, 'fasta')
    SeqIO.write(valid_records, valid_target, 'fasta')

    create_dictionary(download_path, train_records)


def create_dictionary(download_path: str,
                      train_records: Iterable):
    token_counter = Counter()
    for record in train_records:
        token_counter.update(record.seq)

    dictionary = Dictionary()
    for token, count in sorted(token_counter.items()):
        dictionary.add_symbol(token, count)

    with open(os.path.join(download_path, 'dict.txt'), 'w') as f:
        dictionary.save(f)


def main():
    parser = create_parser()
    args = parser.parse_args()

    download_path = os.path.join(args.download_dir, args.dataset)
    os.makedirs(download_path, exist_ok=True)

    try:
        download_command = get_download_commands(args.dataset, download_path)
        print(f'Executing [{download_command}]')
        os.system(download_command)
    except FileExistsError as e:
        print(e)

    split_data(args.dataset, download_path)


if __name__ == '__main__':
    main()
