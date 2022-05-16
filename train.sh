#!/bin/bash
fairseq-train \
  /gscratch/cse/psturm/data/proteins/example/ \
  --user-dir /gscratch/cse/psturm/protein_probabilities \
  --dataset-impl fasta \
  --task masked_lm \
  --criterion masked_lm \
  --arch perceiver \
  --max-tokens 1096 \
  --update-freq 1 \
  --lr 1e-4 \
  --optimizer adam \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 16000 \
  --skip-invalid-size-inputs-valid-test \
  --max-valid-steps 20 \
  --validate-interval-updates 5000 \
  --save-interval-updates 5000;