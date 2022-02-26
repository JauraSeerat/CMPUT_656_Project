from dataclasses import dataclass
import json


@dataclass
class Mention:
    category: str
    text: str
    context_document_id: str
    label_document_id: str
    mention_id: str
    corpus: str
    start_index: int
    end_index: int


class MentionReader:
    def __init__(self, path):
        self.path = path
        self.x = 1

    def __iter__(self):
        with open(self.path, "r") as f:
            for line in f:
                mention_dict = json.loads(line)
                mention = Mention(**mention_dict)
                yield mention

    def read_all(self) -> list[Mention]:
        return [mention for mention in self]
