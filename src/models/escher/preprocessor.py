from src.data.entity import Entity
from src.data.mention import Mention
from src.models.base import BasePreprocessor

from .esc.esc_dataset import DataElement
from .esc.utils.definitions_tokenizer import DefinitionsTokenizer


class EscherPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mention_window_size: int,
        entity_length: int,
        entity_dict: dict[str, Entity],
        tokenizer,
    ) -> None:
        self.mention_window_size = mention_window_size
        self.entity_length = entity_length
        self.entity_dict = entity_dict
        self.tokenizer: DefinitionsTokenizer = tokenizer

    def preprocess_mention(self, mention: Mention) -> str:
        mention_entity = self.entity_dict[mention.context_document_id]
        mention_text = mention_entity.text

        # Tokenize
        mention_tokens = mention_text.split(" ")

        # Insert <classify>, </classify> special tags
        num_target_tokens = mention.end_index - mention.start_index + 1
        mention_tokens.insert(mention.start_index, "<classify>")
        mention_tokens.insert(
            mention.start_index + num_target_tokens + 1, "</classify>"
        )

        # Select tokens within the window size
        window_start = mention.start_index - self.mention_window_size
        window_end = mention.end_index + self.mention_window_size + 3
        mention_tokens = mention_tokens[window_start:window_end]

        mention_text = " ".join(mention_tokens)

        return mention_text

    def preprocess_entity(self, entity: Entity) -> str:
        entity_tokens = entity.text.split(" ")
        entity_tokens = entity_tokens[: self.entity_length]

        entity_text = " ".join(entity_tokens)

        return entity_text

    def preprocess(
        self, mention: Mention, candidate_entities: list[Entity]
    ) -> DataElement:
        mention_input = self.preprocess_mention(mention)
        candidate_input = [
            self.preprocess_entity(entity) for entity in candidate_entities
        ]
        candidate_labels = [
            entity.document_id for entity in candidate_entities
        ]

        (
            encoded_final_sequence,
            candidate_positions,
            token_type_ids,
        ) = self.tokenizer.prepare_sample(mention_input, candidate_input)

        data_element = DataElement(
            encoded_final_sequence=encoded_final_sequence,
            possible_offsets=candidate_labels,
            gloss_positions=candidate_positions,
            token_type_ids=token_type_ids,
        )

        return data_element
