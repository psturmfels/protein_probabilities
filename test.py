import torch.nn as nn
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture
)


@register_model('test')
class TestModel(FairseqEncoderModel):
    @classmethod
    def build_model(cls, args, task):
        print(args)
        print(task)

        encoder = TestEncoder(args, task.dictionary)
        return cls(args, encoder)

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

    def forward(self, src_tokens, segment_labels=None, **kwargs):
        return self.encoder(src_tokens, segment_labels=segment_labels, **kwargs)


class TestEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.x = nn.Linear(in_features=1, out_features=1)
        print(f'Init args: {args}')
        print(f'Init dictionary: {dictionary}')

    def reorder_encoder_out(self, encoder_out, new_order):
        pass

    def forward(self, src_tokens, src_lengths=None, segment_labels=None, masked_tokens=None, **kwargs):
        print(src_tokens)
        print(src_lengths)
        print(segment_labels)
        print(masked_tokens)
        print(kwargs)
        assert 1 == 2


@register_model_architecture('test', 'test')
def base_architecture(args):
    pass
