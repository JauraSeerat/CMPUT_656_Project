import os
from argparse import ArgumentParser
from typing import Dict

import torch
from src.candidate_generators import TfidfCandidateGenerator
from src.data.entity import Entity, EntityReader
from src.data.mention import MentionReader
from src.models.escher.dataset import EscherDataset
from src.models.escher.esc.esc_pl_module import ESCModule
from src.models.escher.esc.predict import PredictionReport, predict
from src.models.escher.preprocessor import EscherPreprocessor
from torch.utils.data import DataLoader

cd = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(cd, "artifacts", "escher_semcor_best.ckpt"),
    )
    parser.add_argument("--mentions_path", type=str, default="data/mentions")
    parser.add_argument("--filename", type=str, default="val.json")
    parser.add_argument("--tokens_per_batch", type=int, default=1024)
    parser.add_argument("--top_k_candidates", type=int, default=16)
    parser.add_argument("--documents_path", type=str, default="data/documents")
    parser.add_argument("--mention_window_size", type=int, default=16)
    parser.add_argument("--entity_length", type=int, default=32)
    parser.add_argument("--prediction_type", type=str, default="probabilistic")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    return args


def get_dataloader(
    mentions_path: str,
    filename: str,
    tokens_per_batch: int,
    entity_dict: Dict[str, Entity],
    preprocessor: EscherPreprocessor,
    re_init_on_iter: bool = False,
    is_test: bool = False,
    top_k: int = 64,
) -> DataLoader:
    mention_reader = MentionReader(os.path.join(mentions_path, filename))
    candidate_generator = TfidfCandidateGenerator(
        entity_dict=entity_dict, top_k=top_k, filename=filename
    )

    dataset = EscherDataset(
        tokens_per_batch=tokens_per_batch,
        re_init_on_iter=re_init_on_iter,
        candidate_generator=candidate_generator,
        is_test=is_test,
        mention_reader=mention_reader,
        preprocessor=preprocessor,
    )
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=None, num_workers=0
    )

    return dataloader


def get_accuracy(
    prediction_report: PredictionReport, mention_num: int, normalized=False
):
    tp = 0
    total = 0

    for instance_prediction in prediction_report.instances_prediction_reports:
        prediciton = instance_prediction.predicted_synsets[0]
        gold = instance_prediction.gold_synsets[0]

        if prediciton == gold:
            tp += 1

        total += 1

    total = total if normalized else mention_num
    accuracy = tp / total

    return accuracy


def main():
    args = parse_args()

    model = ESCModule.load_from_checkpoint(args.ckpt_path)
    model.freeze()

    if args.device >= 0:
        model.to(torch.device(args.device))

    entity_reader = EntityReader(args.documents_path)
    entity_dict = entity_reader.read_all()
    preprocessor = EscherPreprocessor(
        mention_window_size=args.mention_window_size,
        entity_length=args.entity_length,
        entity_dict=entity_dict,
    )

    dataloader = get_dataloader(
        mentions_path=args.mentions_path,
        filename=args.filename,
        tokens_per_batch=args.tokens_per_batch,
        entity_dict=entity_dict,
        preprocessor=preprocessor,
        re_init_on_iter=False,
        is_test=True,
        top_k=args.top_k_candidates,
    )

    prediction_report = predict(
        model=model,
        data_loader=dataloader,
        device=args.device,
        prediction_type=args.prediction_type,
        evaluate=True,
    )

    mention_reader = MentionReader(
        os.path.join(args.mentions_path, args.filename)
    )
    mention_num = len(mention_reader.read_all())

    accuracy = get_accuracy(
        prediction_report=prediction_report,
        mention_num=mention_num,
        normalized=False,
    )

    normalized_accuracy = get_accuracy(
        prediction_report=prediction_report,
        mention_num=mention_num,
        normalized=True,
    )

    print(f"Accuracy: {accuracy * 100:.3f}")
    print(f"Normalizaed accuracy: {normalized_accuracy * 100:.3f}")


if __name__ == "__main__":
    main()
