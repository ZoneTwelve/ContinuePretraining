import lightning as L
from torchmetrics import Metric

from ...metrics import Perplexity
from ..hf import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class LightningModuleForPerplexity(L.LightningModule):
    def __init__(
        self,
        model_path: str,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=None)
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        if max_length is not None:
            if self.config.model_type == 'mpt':
                self.config.max_seq_len = max_length
            elif self.config.model_type == 'llama':
                if max_length is not None:
                    self.config.rope_scaling = {
                        'type': 'dynamic',
                        'factor': max_length / self.config.max_position_embeddings
                    }

        self.ppl = Perplexity(ignore_index=-100)

    def configure_sharded_model(self) -> None:
        kwargs = {
            'torch_dtype': 'auto',
            'low_cpu_mem_usage': True,
            'config': self.config
        }

        from ...lightning import DeepSpeedStrategy
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            kwargs['low_cpu_mem_usage'] = not self.trainer.strategy.zero_stage_3 

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

        for m in self.modules():
            if isinstance(m, Metric):
                m.to(self.strategy.root_device)

    def test_step(self, batch, batch_idx: int):
        x = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        if x.logits.isnan().any():
            return

        self.log('ppl', self.ppl(x.loss, batch['labels']), prog_bar=True, logger=False, on_step=True)
        self.log('PPL/Test', self.ppl, batch_size=batch['input_ids'].size(0), sync_dist=True)
