import json
import os
from data.entity import Entity
from data.mention import Mention
from candidate_generators.tfidfcandidate import Candidates, CandidateReader

from candidate_generators.base import BaseCandidateEntityGenerator

class MyGeneration(BaseCandidateEntityGenerator):
    def __init__(self, entity_dict: dict[str, Entity], path, filename):
        BaseCandidateEntityGenerator.__init__(self, entity_dict)
        self.path = path
        self.filename = filename

    def generate(self, mention: Mention, index) -> list[Entity]:
        mention_id = Mention.mention_id
        candidates = self.get_tfidfData(self.path, self.filename)
        if mention_id in candidates.keys():
            tfidfcandidates = candidates[mention_id]
        entities = list()
        for cand in tfidfcandidates:
            entities.append(self.entity_dict[cand])
        return entities



    def get_tfidfData(self, path, filename) -> dict[str, list[str]]:
        returning_dict = dict()
        with open(os.path.join(self.path, filename), "r") as f:
            for line in f:
                candidate_dict = json.loads(line)
                returning_dict[candidate_dict['mention_id']] = candidate_dict['tfidf_candidates']

        return returning_dict