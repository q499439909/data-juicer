from dsh_demo_token import text_signature

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("demo_text_signature_mapper")
class DemoTextSignatureMapper(Mapper):
    """H1 fixture operator supplied by a derived, model-free capability image."""

    def __init__(self, output_key: str = "demo_signature", **kwargs):
        super().__init__(**kwargs)
        self.output_key = output_key

    def process_single(self, sample):
        sample[self.output_key] = text_signature(sample[self.text_key])
        return sample
