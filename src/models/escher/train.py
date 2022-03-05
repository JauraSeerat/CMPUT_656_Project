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

    args = parser.parse_args()

    return args


def download_artifacts():
    # Escher checkpoint
    url = "https://drive.google.com/uc?id=100jxjLIdmSzrMXXOWgrPz93EG0JBnkfr"

    os.makedirs(os.path.join(cd, "artifacts"), exist_ok=True)
    checkpoint_path = os.path.join(cd, "artifacts", "escher_semcor_best.ckpt")

    gdown.cached_download(url, checkpoint_path)


def get_dataloader(
    datapath: str,
    candidate_filename: str,
    tokens_per_batch: int,
    entity_dict: dict[str, Entity],
    preprocessor: EscherPreprocessor,
    re_init_on_iter: bool = False,
    is_test: bool = False,
    top_k: int = 64,
) -> DataLoader:
    mention_reader = MentionReader(datapath)
    candidate_generator = TfidfCandidateGenerator(
        entity_dict=entity_dict, top_k=top_k, filename=candidate_filename
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

    model = ESCModule.load_from_checkpoint(
        os.path.join(cd, "artifacts", "escher_semcor_best.ckpt")
    )

    entity_reader = EntityReader("data/documents")
    entity_dict = entity_reader.read_all()
    preprocessor = EscherPreprocessor(
        mention_window_size=16, entity_length=32, entity_dict=entity_dict
    )

    train_dataloader = get_dataloader(
        datapath="data/mentions/train.json",
        candidate_filename="train.json",
        tokens_per_batch=1024,
        entity_dict=entity_dict,
        preprocessor=preprocessor,
        re_init_on_iter=False,
        is_test=False,
        top_k=16,
    )

    val_dataloader = get_dataloader(
        datapath="data/mentions/val.json",
        candidate_filename="val.json",
        tokens_per_batch=1024,
        entity_dict=entity_dict,
        preprocessor=preprocessor,
        re_init_on_iter=False,
        is_test=False,
        top_k=16,
    )

    model_checkpoint = ModelCheckpoint(save_top_k=0, verbose=True, mode="max")
    wandb_logger = WandbLogger(project="escher")

    trainer = pl.Trainer(
        gpus=0,
        accumulate_grad_batches=20,
        gradient_clip_val=10.0,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint,
        max_steps=2,
        val_check_interval=200,
        weights_save_path=os.path.join(cd, "checkpoint"),
    )

    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return args


if __name__ == "__main__":
    main()
