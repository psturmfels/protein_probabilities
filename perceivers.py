from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from transformers import PerceiverConfig, PerceiverForMaskedLM, PerceiverTokenizer


@register_model('perceiver')
class PerceiverEncoderModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

    def forward(self, src_tokens, **kwargs):
        return self.encoder(src_tokens, **kwargs)

    def max_positions(self):
        return self.encoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        '''Build a new model instance.'''
        # make sure all arguments are present in older models
        base_architecture(args)

        encoder = PerceiverEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    @staticmethod
    def add_args(parser, **kwargs):
        parser.add_argument(
            '--num_latents',
            type=int,
            metavar='N',
            help='The number of latent variables',
        )
        parser.add_argument(
            '--d_latents',
            type=int,
            metavar='N',
            help='The dimension of the latent variables',
        )
        parser.add_argument(
            '--d_model',
            type=int,
            metavar='N',
            help='The dimension of the model inputs',
        )
        parser.add_argument(
            '--num_blocks',
            type=int,
            metavar='N',
            help='The number of blocks in the self-encoder',
        )
        parser.add_argument(
            '--num_self_attends_per_block',
            type=int,
            metavar='N',
            help='The number of self-attention layers per block',
        )
        parser.add_argument(
            '--num_self_attention_heads',
            type=int,
            metavar='N',
            help='Number of attention heads for each self-attention layer in the Transformer encoder',
        )
        parser.add_argument(
            '--num_cross_attention_heads',
            type=int,
            metavar='N',
            help='Number of attention heads for each cross-attention layer in the Transformer encoder',
        )


class PerceiverEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.config = PerceiverConfig(**args)
        self.model = PerceiverForMaskedLM(config=self.config)

    def forward(
            self,
            src_tokens,
            masked_tokens=None,
            **unused,
    ):
        outputs = self.model(inputs=src_tokens, attention_mask=masked_tokens)
        logits = outputs.logits
        return logits, outputs


@register_model_architecture('perceiver', 'perceiver')
def base_architecture(args):
    args.num_latents = getattr(args, 'num_latents', 256)
    args.d_latents = getattr(args, 'd_latents', 1280)
    args.d_model = getattr(args, 'd_model', 768)
    args.num_blocks = getattr(args, 'num_blocks', 1)
    args.num_self_attends_per_block = getattr(args, 'num_self_attends_per_block', 26)
    args.num_self_attention_heads = getattr(args, 'num_self_attention_heads', 8)
    args.num_cross_attention_heads = getattr(args, 'num_cross_attention_heads', 8)


def main():
    tokenizer = PerceiverTokenizer()
    config = PerceiverConfig(num_latents=16,
                             d_latents=32,
                             d_model=32,
                             num_blocks=1,
                             num_self_attends_per_block=8)
    model = PerceiverForMaskedLM(config=config)

    text = 'This is an incomplete sentence where some words are missing.'
    inputs = tokenizer(text, padding='max_length', return_tensors='pt')
    print(inputs)
    outputs = model(**inputs)
    logits = outputs.logits
    print(logits)


if __name__ == '__main__':
    main()
