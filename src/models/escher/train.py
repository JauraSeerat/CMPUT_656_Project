import os
from argparse import ArgumentParser

import gdown
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.candidate_generators import TfidfCandidateGenerator
from src.data.entity import Entity, EntityReader
from src.data.mention import MentionReader
from src.models.escher.dataset import EscherDataset
from src.models.escher.esc.esc_pl_module import ESCModule
from src.models.escher.preprocessor import EscherPreprocessor
from torch.utils.data import DataLoader

cd = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--documents_path", type=str, default="data/documents")
    parser.add_argument("--mentions_path", type=str, default="data/mentions")
    parser.add_argument("--mention_window_size", type=int, default=16)
    parser.add_argument("--entity_length", type=int, default=32)
    parser.add_argument("--top_k_candidates", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--tokens_per_batch", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=10.0)
    parser.add_argument("--val_check_interval", type=int, default=200)
    parser.add_argument("--save_top_k_ckpts", type=int, default=0)
    parser.add_argument(
        "--weights_save_path", type=str, default=os.path.join(cd, "checkpoint")
    )
    parser.add_argument(
        "--escher_ckpt",
        type=str,
        default=os.path.join(cd, "artifacts", "escher_semcor_best.ckpt"),
    )
    parser.add_argument("--wandb_project", type=str, default="escher")

    args = parser.parse_args()

    return args


def download_artifacts():
    # Escher checkpoint
    url = "https://drive.google.com/uc?id=100jxjLIdmSzrMXXOWgrPz93EG0JBnkfr"

    os.makedirs(os.path.join(cd, "artifacts"), exist_ok=True)
    checkpoint_path = os.path.join(cd, "artifacts", "escher_semcor_best.ckpt")

    gdown.cached_download(url, checkpoint_path)


def get_dataloader(
    mentions_path: str,
    filename: str,
    tokens_per_batch: int,
    entity_dict: dict[str, Entity],
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


def main():
    args = parse_args()

    download_artifacts()

    model = ESCModule.load_from_checkpoint(args.escher_ckpt)

    entity_reader = EntityReader(args.documents_path)
    entity_dict = entity_reader.read_all()
    preprocessor = EscherPreprocessor(
        mention_window_size=args.mention_window_size,
        entity_length=args.entity_length,
        entity_dict=entity_dict,
    )

    train_dataloader = get_dataloader(
        mentions_path=args.mentions_path,
        filename="train.json",
        tokens_per_batch=args.tokens_per_batch,
        entity_dict=entity_dict,
        preprocessor=preprocessor,
        re_init_on_iter=True,
        is_test=False,
        top_k=args.top_k_candidates,
    )

    val_dataloader = get_dataloader(
        mentions_path=args.mentions_path,
        filename="val.json",
        tokens_per_batch=args.tokens_per_batch,
        entity_dict=entity_dict,
        preprocessor=preprocessor,
        re_init_on_iter=False,
        is_test=True,
        top_k=args.top_k_candidates,
    )

    model_checkpoint = ModelCheckpoint(
        save_top_k=args.save_top_k_ckpts, verbose=True, mode="max"
    )
    wandb_logger = WandbLogger(project=args.wandb_project)

    trainer = pl.Trainer(
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        weights_save_path=args.weights_save_path,
    )

    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return args


if __name__ == "__main__":
    main()
