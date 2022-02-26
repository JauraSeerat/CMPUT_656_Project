from abc import ABC, abstractmethod
from typing import Any

from data.entity import Entity
from data.mention import Mention


class BaseModel(ABC):
    pass


class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, mention: Mention, candidate_entities: list[Entity]
    ) -> Any:
        pass
