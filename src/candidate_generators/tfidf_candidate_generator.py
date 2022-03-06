import json
import os
import tarfile
from typing import Dict, List

import gdown
from src.candidate_generators.base import BaseCandidateEntityGenerator
from src.data.entity import Entity
from src.data.mention import Mention

cd = os.path.dirname(os.path.abspath(__file__))


class TfidfCandidateGenerator(BaseCandidateEntityGenerator):
    def __init__(
        self,
        entity_dict: Dict[str, Entity],
        top_k: int = 64,
        filename: str = "train.json",
        path: str = os.path.join(cd, "artifacts", "tfidf_candidates"),
    ):
        BaseCandidateEntityGenerator.__init__(self, entity_dict)
        self.path = path
        self.filename = filename
        self.top_k = top_k
        self.download_artifacts()
        self.candidate_dict = self.get_candidate_dict()

    def generate(self, mention: Mention) -> List[Entity]:
        mention_id = mention.mention_id

        if mention_id not in self.candidate_dict:
            return []

        candidate_ids = self.candidate_dict[mention_id]
        candidate_entities = [
            self.entity_dict[candidate_id] for candidate_id in candidate_ids
        ]

        return candidate_entities[: self.top_k]

    def get_candidate_dict(self) -> Dict[str, List[str]]:
        candidate_dict = {}
        with open(os.path.join(self.path, self.filename), "r") as f:
            for line in f:
                line_dict = json.loads(line)
                candidate_dict[line_dict["mention_id"]] = line_dict[
                    "tfidf_candidates"
                ]

        return candidate_dict

    def download_artifacts(self):
        if os.path.isfile(os.path.join(self.path, self.filename)):
            return

        url = (
            "https://drive.google.com/u/0/uc?"
            "id=1wGppj3ivE7jBaDzDlovWAvaBzLhPjR8B&export=download"
        )

        archive_path = self.path + ".tar.bz2"

        os.makedirs(self.path, exist_ok=True)
        gdown.cached_download(url, archive_path)

        with tarfile.open(archive_path, "r:bz2") as f:
            f.extractall(os.path.dirname(archive_path))

        os.remove(archive_path)
