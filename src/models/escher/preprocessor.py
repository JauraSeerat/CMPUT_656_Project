from data.entity import Entity
from data.mention import Mention
from models.base import BasePreprocessor
from models.escher.esc.esc_dataset import DataElement
from models.escher.esc.utils.definitions_tokenizer import DefinitionsTokenizer


class EscherPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mention_length: int,
        entity_length: int,
        entity_dict: dict[str, Entity],
        tokenizer: DefinitionsTokenizer,
    ) -> None:
        self.mention_length = mention_length
        self.entity_length = entity_length
        self.entity_dict = entity_dict
        self.tokenizer = tokenizer

    def annotate(self, mention: Mention, mention_text: str) -> str:
        return mention_text

    def preprocess_mention(self, mention: Mention) -> str:
        mention_entity = self.entity_dict[mention.context_document_id]
        mention_text = mention_entity.text
        mention_text = self.annotate(mention, mention_text)

        return mention_text

    def preprocess_entity(self, entity: Entity) -> str:
        return entity.text

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
