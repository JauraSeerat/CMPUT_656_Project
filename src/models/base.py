from abc import ABC, abstractmethod
from typing import Any, List

from src.data.entity import Entity
from src.data.mention import Mention


class BaseModel(ABC):
    pass


class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, mention: Mention, candidate_entities: List[Entity]
    ) -> Any:
        pass
