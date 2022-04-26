#!/usr/bin/env bash
pip install numpy pandas six tape-proteins matplotlib altair
pip install jaxlib
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install fair-esm
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
svn export https://github.com/google-research/google-research/trunk/protein_lm