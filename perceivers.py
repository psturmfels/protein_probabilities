from transformers import PerceiverConfig, PerceiverForMaskedLM
from fairseq.models import FairseqEncoder

def PerceiverEncoder():


def main():
    config = PerceiverConfig(num_latents=16,
                             d_latents=32,
                             d_model=32,
                             num_blocks=1,
                             num_self_attends_per_block=8,
                             vocab_size=23,
                             max_position_embeddings=128)
    model = PerceiverForMaskedLM(config=config)
    model.to('cuda')
    print(model)

if __name__  == '__main__':
    main()