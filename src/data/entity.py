from dataclasses import dataclass
import os
import json
from typing import Dict


@dataclass
class Entity:
    title: str
    text: str
    document_id: str


class EntityReader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for filename in os.listdir(self.path):
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    entity_dict = json.loads(line)
                    entity = Entity(**entity_dict)
                    yield entity

    def read_all(self) -> Dict[str, Entity]:
        return {entity.document_id: entity for entity in self}
