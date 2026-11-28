import torch
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence


def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):

    tokenizer_pad_size = 'right'
    tokenizer_pad_id = 1

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)


    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer_pad_id)
    for i, example in enumerate(examples):
        if tokenizer_pad_size == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def get_special_tokens_mask( token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
) -> List[int]:

    all_special_ids = [0,1,2,3,4]

    special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

    return special_tokens_mask


class CustomDataCollatorForLanguageModeling:

    tokenizer: Tokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tokenizer_mask_id = 4
    tokenizer_pad_id = 1

    def __init__(self,tokenizer,mlm,mlm_probability):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        special_tokens_mask = batch.pop("special_tokens_mask", None)

        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer_pad_id is not None:
                labels[labels == self.tokenizer_pad_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer_mask_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels