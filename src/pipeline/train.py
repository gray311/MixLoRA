import os
os.environ["WANDB_PROJECT"] = "mixlora"  #

import sys
import logging
import pathlib
import typing
import warnings

project_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_path))

import json
from tqdm import tqdm
from dataclasses import field, dataclass
from typing import Optional, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from dataclasses import dataclass, field

import evaluate
import transformers
from transformers import Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq, DataCollatorWithPadding, EvalPrediction
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration, PreTrainedTokenizer, BatchEncoding

from dataset import Seq2SeqDataset, Seq2SeqCollator

from peft import (
    MixLoraConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List, Sequence, Dict, Any
import logging

from config.lora_config import mixlora_config, lora_config
from pipeline.utils import load_checkpoint
from model.moe_t5 import T5ForConditionalGeneration, T5ForSequenceClassification
from model.config import T5MoEConfig
from pipeline.trainer import TextToTextTrainer


logging.basicConfig(level=logging.INFO)

glue_tasks = [
    "cola",
    "mnli-m",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli-m": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_id2label = {
    "cola": {0: 0, 1: 1},
    "mnli_matched": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "mnli_mismatched": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "mrpc": {0: 0, 1: 1},
    "qnli": {0: 'entailment', 1: 'not_entailment'},
    "qqp": {0: 0, 1: 1},
    "rte": {0: 'entailment', 1: 'not_entailment'},
    "sst2": {0: 0, 1: 1},
    "wnli": {0: 0, 1: 1},
    "stsb": {0.0: 0.0}
}

task_to_instructions = {
    "rte": "Determine if the second sentence is logically entailed by, or contradicts, the first sentence.\nFirst Sentence: {sentence1}\nSecond sentence: {sentence2}\n\nAnswer: ",
    "mrpc": "Determine if the two sentences are paraphrases of each other.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer: ",
    "stsb": "Rate the semantic similarity of the two sentences on a scale from 0 to 5.\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer: ",
    "cola": "Judge whether the following sentence is grammatically correct and linguistically acceptable.\nSentence: {sentence}\n\nAnswer: "
}

@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")
    data_paths: List[str] = field(default_factory=lambda: ["./alpaca_data.json"], metadata={"help": "Path to the training data."})
    data_name: str = field(default="alpaca")
    instruction_length: int = 256
    output_length: int = 5
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    use_peft: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

class TextToTextSample(BaseModel):
    source: str
    target: str
    
class TextToLabelSample(BaseModel):
    source: str
    target: int

class TextToScoreSample(BaseModel):
    source: str
    target: float

class TextToTextData(BaseModel):
    samples: List[TextToTextSample]

    @classmethod
    def load(cls, path: str, data_name: str, mode: str):
        if data_name == "alpaca":
            with open(path) as f:
                all_lines = json.load(f)
                samples = []
                for line in all_lines:
                    source = line['instruction'].strip()
                    if line['input'].strip():
                        source = source + "\n" + line['input']
                    target = line['output']
                    samples.append(TextToTextSample(**{'source': source, 'target': target}))

        elif data_name == "mmlu":
            subjects = sorted(
                [
                    f.split(".csv")[0]
                    for f in os.listdir(os.path.join(path, "auxiliary_train"))
                    if ".csv" in f
                ]
            )

            train_df = pd.DataFrame()
            for subject in tqdm(subjects):
                tmp_df = pd.read_csv(
                    os.path.join(path, "auxiliary_train", subject + ".csv"), header=None
                )
                train_df = pd.concat([train_df, tmp_df])

            def get_choices():
                return ["A", "B", "C", "D"]

            def format_example(df, idx, include_answer=True):
                prompt = df.iloc[idx, 0]
                k = df.shape[1] - 2
                for j in range(k):
                    prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
                prompt += "\nAnswer:"
                answer = ""
                if include_answer:
                    answer += " {}\n\n".format(df.iloc[idx, k + 1])
                return prompt, answer

            samples = []
            for i in range(train_df.shape[0]):
                source, target = format_example(train_df, i)
                samples.append(TextToTextSample(**{'source': source, 'target': target}))
                
        elif data_name in glue_tasks:
            key1, key2 = task_to_keys[data_name]
            id2label = task_to_id2label[data_name]
            
            import pandas
            train_df = pd.read_csv(os.path.join(path, f"{mode}.csv"))
            
            samples = []
            for i in range(train_df.shape[0]):
                if key2 is not None:
                    sentence1, sentence2, label = train_df.iloc[i, :3]
                    source = task_to_instructions[data_name].format(sentence1=sentence1, sentence2=sentence2)
                else:
                    sentence, label = train_df.iloc[i, :2]
                    source = task_to_instructions[data_name].format(sentence=sentence)
                
                if data_name == "stsb":
                    samples.append(TextToScoreSample(**{'source': source, 'target': label}))
                else:
                    samples.append(TextToLabelSample(**{'source': source, 'target': label}))
                    
        return cls(samples=samples)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for sample in self.samples:
                print(sample.json(), file=f)

    def analyze(self, num: int = 10, tokenizer_name: str = "t5-base"):
        random.seed(num)
        for sample in random.sample(self.samples, k=num):
            print(sample.json(indent=2))

        token_checker = TokensLengthAnalyzer(name=tokenizer_name)
        info = dict(
            total_samples=len(self.samples),
            source=str(token_checker.run([sample.source for sample in self.samples])),
            target=str(token_checker.run([sample.target for sample in self.samples])),
        )
        print(json.dumps(info, indent=2))

class TextToTextDataset(Dataset):
    def __init__(
        self,
        data_name: str,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int,
        max_target_length: int,
        mode: str,
    ):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.data = TextToTextData.load(path, data_name, mode)

    def __len__(self) -> int:
        return len(self.data.samples)

    def tokenize(self, text: str, is_source: bool) -> BatchEncoding:
        x = self.tokenizer(
            text,
            max_length=self.max_source_length if is_source else self.max_target_length,
            padding="max_length",
            truncation=not is_source,
            return_tensors="pt",
        )

        """
        T5 truncates on right by default, but we can easily truncate on left
        for the encoder input as there is no special token on the left side
        """
        if is_source:
            assert x.input_ids.ndim == 2
            assert x.input_ids.shape == x.attention_mask.shape
            length = x.input_ids.shape[1]
            start = max(length - self.max_source_length, 0)
            x.input_ids = x.input_ids[:, start:]
            x.attention_mask = x.attention_mask[:, start:]
            assert x.input_ids.shape[1] == self.max_source_length

        return x

    def __getitem__(self, i: int) -> dict:
        if isinstance(self.data.samples[i].target, torch.Tensor):
            x = self.tokenize(self.data.samples[i].source, is_source=True)
            y = self.tokenize(self.data.samples[i].target, is_source=False)
            return {
                "input_ids": x.input_ids.squeeze(),
                "attention_mask": x.attention_mask.squeeze(),
                "labels": y.input_ids.squeeze(),
                "decoder_attention_mask": y.attention_mask.squeeze(),
            }
            
        else:
            x = self.tokenize(self.data.samples[i].source, is_source=True)
            if self.data_name == "stsb":
                y = torch.tensor([float(self.data.samples[i].target)], dtype=torch.float16)
            else:
                y = torch.tensor([int(self.data.samples[i].target)], dtype=torch.long)
            return {
                "input_ids": x.input_ids.squeeze(),
                "attention_mask": x.attention_mask.squeeze(),
                "labels": y,
            }
        

    def to_human_readable(self, raw: dict) -> dict:
        source = self.tokenizer.decode(raw["source_ids"])
        target = self.tokenizer.decode(raw["target_ids"])
        return dict(source=source, target=target)


# @dataclass
# class DataCollatorForLMDataset(object):
#     """Collate examples for supervised fine-tuning."""
#
#     tokenizer: transformers.PreTrainedTokenizer
#     IGNORE_INDEX: int
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids = [instance['input_ids'] for instance in instances]
#         attention_mask = [instance['attention_mask'] for instance in instances]
#         decoder_attention_mask = [instance['decoder_attention_mask'] for instance in instances]
#         labels = [instance['labels'] for instance in instances]
#
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=self.tokenizer.pad_token_id)
#
#         labels[labels[:, :] == self.tokenizer.pad_token_id] = self.IGNORE_INDEX
#
#         batch = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#             decoder_attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )
#
#         return batch

def train():
    parser = transformers.HfArgumentParser(Seq2SeqTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model_name = args.model_name_or_path.split("/")[-1].strip(" ")
    args.output_dir = os.path.join(args.output_dir, model_name, args.data_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )
    
    if args.data_name in glue_tasks:
        num_labels = len(task_to_id2label[args.data_name].keys())
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir,
        )
        config.num_labels=num_labels
        config.classifier_dropout=0.01
        model = T5ForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            use_cache=False,
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
        )

    if args.use_peft is not None:
        args.output_dir = os.path.join(args.output_dir, args.use_peft)
        if args.use_peft == "mixlora":
            config = MixLoraConfig(**mixlora_config)
        elif args.use_peft == "lora":
            config = LoraConfig(**lora_config)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    for n, p in model.named_parameters():
        if p.requires_grad or "lora_" in n or "classification_":
            p.requires_grad = True
            print(n, p.shape)
 
    if args.data_name in glue_tasks:
        train_dataset = TextToTextDataset(
            data_name=args.data_name,
            path=args.data_paths[0],
            tokenizer=tokenizer,
            max_source_length=args.instruction_length,
            max_target_length=args.output_length,
            mode="train",
        )
        eval_dataset = TextToTextDataset(
            data_name=args.data_name,
            path=args.data_paths[0],
            tokenizer=tokenizer,
            max_source_length=args.instruction_length,
            max_target_length=args.output_length,
            mode="dev",
        )
        
        is_regression = args.data_name == "stsb"
        
        metric = evaluate.load("glue", args.data_name)
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8 if args.fp16 else None,
        )

    else:
        train_dataset = TextToTextDataset(
            data_name=args.data_name,
            path=args.data_paths[0],
            tokenizer=tokenizer,
            max_source_length=args.instruction_length,
            max_target_length=args.output_length,
            mode="train",
        )
        eval_dataset, compute_metrics = None, None
        
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if args.fp16 else None,
        )

    trainer = TextToTextTrainer(
        model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
