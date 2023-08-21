from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from taide_cp.data.datamodule import StageType

from ...utils import DatasetsContextManager
from ..datamodule import LightningDataModuleX
from .datacollator_for_multiple_choice_question import \
    DataCollatorForMultipleChoiceQuestion


def preprocess(x, i):
    x['id'] = i
    x['question'] = x.pop('題目')
    x['choices'] = [
        x.pop('選項一'),
        x.pop('選項二'),
        x.pop('選項三'),
        x.pop('選項四').removesuffix('。')
    ]
    x['answer'] = x.pop('正確答案') - 1
    return x

class DataModuleForMultipleChoiceQuestion(LightningDataModuleX):
    datacollator_cls = DataCollatorForMultipleChoiceQuestion
    datacollator: DataCollatorForMultipleChoiceQuestion
    default_split = 'test'

    def __init__(self, tokenizer: PreTrainedTokenizerBase, k_shot: int, convert_to_chs: bool = False, *, data_path: str | None = None, dataset_path: str | None = None, train_batch_size: int = 1, val_batch_size: int | None = None, test_batch_size: int | None = None, pred_batch_size: int | None = None, train_split_size: int | float | None = None, val_split_size: int | float | None = None, test_split_size: int | float | None = None, pred_split_size: int | float | None = None, split_seed: int | None = 42, num_workers: int = 1, pin_memory: bool = True, num_proc: int | None = None) -> None:
        datacollator_kwargs = dict(convert_to_chs=convert_to_chs)
        super().__init__(tokenizer, data_path=data_path, dataset_path=dataset_path, train_batch_size=train_batch_size, val_batch_size=val_batch_size, test_batch_size=test_batch_size, pred_batch_size=pred_batch_size, train_split_size=train_split_size, val_split_size=val_split_size, test_split_size=test_split_size, pred_split_size=pred_split_size, split_seed=split_seed, num_workers=num_workers, pin_memory=pin_memory, num_proc=num_proc, datacollator_kwargs=datacollator_kwargs)

        self.k_shot = k_shot

    @DatasetsContextManager(progress_bar=True)
    def prepare_data(self) -> None:
        if self.is_dataset(self.dataset_path):
            return
        
        dataset = load_dataset(self.data_path)['train']
        dataset = dataset.map(preprocess, with_indices=True)
        dataset.save_to_disk(self.dataset_path)

    def setup(self, stage: StageType | None = None) -> None:
        super().setup(stage)

        dataset = self.dataset['test'].shuffle(seed=42)
        examples = dataset.select(range(self.k_shot))
        self.dataset['test'] = dataset.select(range(self.k_shot, dataset.num_rows))
        self.datacollator.set_examples(examples)
