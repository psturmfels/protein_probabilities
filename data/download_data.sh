#!/usr/bin/env bash
wget http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz
tar xzf fluorescence.tar.gz

wget http://s3.amazonaws.com/proteindata/data_pytorch/stability.tar.gz
tar xzf stability.tar.gz

wget http://s3.amazonaws.com/proteindata/data_pytorch/remote_homology.tar.gz
tar xzf remote_homology.tar.gz

curl -O https://dl.fbaipublicfiles.com/fair-esm/examples/P62593.fasta
