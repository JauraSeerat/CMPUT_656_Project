import json
import os

from data.entity import Entity
from data.mention import Mention

from candidate_generators.base import BaseCandidateEntityGenerator

cd = os.path.dirname(os.path.abspath(__file__))


class MyGeneration(BaseCandidateEntityGenerator):
    def __init__(self, entity_dict: dict[str, Entity], path, filename):
        BaseCandidateEntityGenerator.__init__(self, entity_dict)
        self.candidate_dict = self.get_candidate_dict(path, filename)

    def generate(self, mention: Mention) -> list[Entity]:
        mention_id = mention.mention_id

        if mention_id not in self.candidate_dict:
            return []

        candidate_ids = self.candidate_dict[mention_id]
        candidate_entities = [
            self.entity_dict[candidate_id] for candidate_id in candidate_ids
        ]

        return candidate_entities

    def get_candidate_dict(self, path, filename) -> dict[str, list[str]]:
        candidate_dict = {}
        with open(os.path.join(path, filename), "r") as f:
            for line in f:
                candidate_dict.update(json.loads(line))

        return candidate_dict
