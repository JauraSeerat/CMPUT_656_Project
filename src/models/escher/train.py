from argparse import ArgumentParser
import gdown
import os
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from src.models.escher.dataset import EscherDataset
from src.models.escher.esc.esc_pl_module import ESCModule
from src.models.escher.preprocessor import EscherPreprocessor
from src.data.entity import EntityReader
from src.data.mention import MentionReader
from src.candidate_generators import TfidfCandidateGenerator

cd = os.path.dirname(os.path.abspath(__file__))


def download_artifacts():
    # Escher checkpoint
    url = "https://drive.google.com/uc?id=100jxjLIdmSzrMXXOWgrPz93EG0JBnkfr"

    os.makedirs(os.path.join(cd, "artifacts"), exist_ok=True)
    checkpoint_path = os.path.join(cd, "artifacts", "escher_semcor_best.ckpt")

    gdown.cached_download(url, checkpoint_path)


def main():
    parser = ArgumentParser()

    args = parser.parse_args()

    download_artifacts()

    model = ESCModule.load_from_checkpoint(
        os.path.join(cd, "artifacts", "escher_semcor_best.ckpt")
    )

    entity_reader = EntityReader("data/documents")
    entity_dict = entity_reader.read_all()
    preprocessor = EscherPreprocessor(
        mention_window_size=16, entity_length=32, entity_dict=entity_dict
    )

    train_mention_reader = MentionReader("data/mentions/train.json")
    train_candidate_generator = TfidfCandidateGenerator(
        entity_dict=entity_dict, top_k=16, filename="train.json"
    )

    train_dataset = EscherDataset(
        tokens_per_batch=1024,
        re_init_on_iter=False,
        candidate_generator=train_candidate_generator,
        is_test=False,
        mention_reader=train_mention_reader,
        preprocessor=preprocessor,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, num_workers=0
    )

    val_mention_reader = MentionReader("data/mentions/val.json")
    val_candidate_generator = TfidfCandidateGenerator(
        entity_dict=entity_dict, top_k=16, filename="val.json"
    )

    val_dataset = EscherDataset(
        tokens_per_batch=1024,
        re_init_on_iter=False,
        candidate_generator=val_candidate_generator,
        is_test=False,
        mention_reader=val_mention_reader,
        preprocessor=preprocessor,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    model_checkpoint = ModelCheckpoint(save_top_k=0, verbose=True, mode="max")
    wandb_logger = WandbLogger(project="escher")

    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=20,
        gradient_clip_val=10.0,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint,
        max_steps=10000,
        val_check_interval=200,
        weights_save_path=os.path.join(cd, "checkpoint"),
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return args


if __name__ == "__main__":
    main()
