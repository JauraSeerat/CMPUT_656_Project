from abc import ABC, abstractmethod
from typing import Dict, List

from src.data.entity import Entity
from src.data.mention import Mention


class BaseCandidateEntityGenerator(ABC):
    def __init__(self, entity_dict: Dict[str, Entity]):
        self.entity_dict = entity_dict

    @abstractmethod
    def generate(self, mention: Mention) -> List[Entity]:
        pass
