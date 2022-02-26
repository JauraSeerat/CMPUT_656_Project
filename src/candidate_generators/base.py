from abc import ABC, abstractmethod

from data.entity import Entity
from data.mention import Mention


class CandidateEntityGenerator(ABC):
    @abstractmethod
    def generate(self, mention: Mention) -> list[Entity]:
        pass
