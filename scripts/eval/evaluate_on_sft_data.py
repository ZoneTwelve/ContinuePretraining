import fire

from taide_cp.utils.scripting import *


@entry_point(
    get_logger
)
def main(
    model_path: str,
    data_path: str,
    **kwargs,
):
    import lightning as L

    from taide_cp.data import DataModuleForSupervisedFineTuning
    from taide_cp.training import LightningModuleForPerplexity
    from taide_cp.utils import SLURM
    
    model = LightningModuleForPerplexity(model_path)
    datamodule = DataModuleForSupervisedFineTuning(model.tokenizer, data_path=data_path)
    
    trainer = L.Trainer(
        num_nodes=SLURM.num_nodes or 'auto',
        logger=get_logger('csv', **kwargs),
        enable_checkpointing=False,
    )

    trainer.test(model, datamodule)

if __name__ == '__main__':
    fire.Fire(main)
