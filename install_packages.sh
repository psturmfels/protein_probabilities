#!/usr/bin/env bash
pip install numpy pandas six tape-proteins matplotlib altair scikit-learn fs-gcsfs
pip install jaxlib flax
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install fair-esm
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
svn export https://github.com/google-research/google-research/trunk/protein_lm

pip install gin-config tensorflow
git submodule add git@github.com:googleinterns/protein-embedding-retrieval.git