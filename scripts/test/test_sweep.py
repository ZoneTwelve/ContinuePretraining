import fire
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from taide_cp.data import DataModuleForPreTraining
from taide_cp.lightning import DeepSpeedStrategy
from taide_cp.training import LLaMALightningModuleForPreTraining
from taide_cp.utils.sweep_runner import SweepRunner


def train(
    lr: float,
):
    model = LLaMALightningModuleForPreTraining(
        'checkpoints/llama-7b',
        'checkpoints/tokenizer/llama-v2',
        lr=lr,
        extend_tokens=True
    )
    datamodule = DataModuleForPreTraining('data/cp/tokenized/llama-v2/g', model.tokenizer, num_workers=4)

    trainer = L.Trainer(
        precision='16-mixed',
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            pin_memory=True,
            logging_batch_size_per_gpu=1,
        ),
        logger=WandbLogger(),
        val_check_interval=100,
        limit_val_batches=100,
        max_steps=200,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule)
    trainer.logger.experiment.finish()

def main(
    sweep_id: str,
    count: int,
):
    sweep_runner = SweepRunner()
    sweep_runner.run(sweep_id, train, count)


if __name__ == '__main__':
    fire.Fire(main)
