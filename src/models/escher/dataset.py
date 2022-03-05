from src.candidate_generators.base import BaseCandidateEntityGenerator
from src.data.mention import MentionReader
from src.models.base import BasePreprocessor
from src.models.escher.esc.esc_dataset import QAExtractiveDataset
from src.models.escher.esc.utils.definitions_tokenizer import (
    DefinitionsTokenizer,
)


class EscherDataset(QAExtractiveDataset):
    def __init__(
        self,
        tokenizer: DefinitionsTokenizer,
        tokens_per_batch: int,
        re_init_on_iter: bool,
        preprocessor: BasePreprocessor,
        candidate_generator: BaseCandidateEntityGenerator,
        mention_reader: MentionReader,
        is_test: bool = False,
    ) -> None:
        super().__init__(tokenizer, tokens_per_batch, re_init_on_iter, is_test)

        self.preprocessor = preprocessor
        self.mention_reader = mention_reader
        self.candidate_generator = candidate_generator

    def init_dataset(self):
        self.data_store = []
        for mention in self.mention_reader:
            candidate_entities = self.candidate_generator.generate(mention)
            data_element = self.preprocessor.preprocess(
                mention, candidate_entities
            )
            self.data_store.append(data_element)
