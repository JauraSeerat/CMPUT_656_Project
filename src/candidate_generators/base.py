from abc import ABC, abstractmethod

from data.entity import Entity
from data.mention import Mention


class BaseCandidateEntityGenerator(ABC):
    def __init__(self, entity_dict: dict[str, Entity]):
        self.entity_dict = entity_dict

    @abstractmethod
    def generate(self, mention: Mention) -> list[Entity]:
        pass
