from ...metrics import Perplexity
from ...models import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..lightning_module import LightningModuleX


class LightningModuleForPerplexity(LightningModuleX):
    def __init__(
        self,
        model_path: str,
        max_length: int | None = None,
    ) -> None:
        super().__init__()

        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=None)
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        if max_length is not None:
            if self.config.model_type == 'mpt':
                self.config.max_seq_len = max_length
            elif self.config.model_type == 'llama':
                self.config.max_position_embeddings = max_length

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype='auto', low_cpu_mem_usage=True, config=self.config)

        self.ppl = Perplexity(ignore_index=-100)
        # self.pplu = UnigramNormalizedPerplexity(ignore_index=-100)
        # self.pplk = TopKPerplexity(32000, ignore_index=-100)

    def test_step(self, batch, batch_idx: int):
        x = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],        
        )

        self.log('ppl', self.ppl(x.loss, batch['labels'], by_loss=True), prog_bar=True, logger=False, on_step=True)
        # self.log('pplu', self.pplu(x.logits, batch['labels']), prog_bar=True, logger=False, on_step=True)
        # self.log('pplk', self.pplk(x.logits, batch['labels']), prog_bar=True, logger=False, on_step=True)

        self.log('PPL/Test', self.ppl, batch_size=batch['input_ids'].size(0), sync_dist=True)
        # self.log('PPLu/Test', self.pplu, batch_size=batch['input_ids'].size(0), sync_dist=True)
        # self.log('PPLk/Test', self.pplk, batch_size=batch['input_ids'].size(0), sync_dist=True)
