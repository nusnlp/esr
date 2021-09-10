import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import transformers
import numpy as np
import torch
from importlib import import_module
import logging


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:

    model_name: str = field(
        metadata={'help': 'Model identifier from huggingface.co/models'}
    )
    model_path: Optional[str] = field(
        default=None, metadata={'help': 'Path to pretrained model'}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'}
    )


@dataclass
class DataArguments:

    dataset_index: str = field(
        metadata={'help': 'Dataset used'}
    )
    wsd_xml: str = field(
        metadata={'help': 'Train/Predict xml file'}
    )
    wsd_label: Optional[str] = field(
        default=None,
        metadata={'help': 'Train/Predict label file'}
    )
    dev_xml: Optional[str] = field(
        default=None,
        metadata={'help': 'Dev xml file'}
    )
    dev_label: Optional[str] = field(
        default=None,
        metadata={'help': 'Dev label file'}
    )
    extra_xml: Optional[str] = field(
        default=None,
        metadata={'help': 'Extra xml file'}
    )
    pred_file: Optional[str] = field(
        default=None,
        metadata={'help': 'Predictions in softmax'}
    )
    input_limit: Optional[int] = field(
        default=432,
        metadata={'help': 'Input limit (tokens)'}
    )


@dataclass
class UserTrainingArguments(transformers.TrainingArguments):

    fp32: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use fp32 instead of tf32"}
    )
    early_stopping_patience: Optional[int] =field(
        default=0, metadata={"help": "Number of evaluations before early stopping if no further improvement"}
    )
    save_start_ratio: Optional[float] = field(
        default=0.0, metadata={"help": "Ratio of total training steps before save start"}
    )
    save_start_steps: Optional[int] = field(
        default=0, metadata={"help": "Number of training steps before save start"}
    )


class UserFlowCallback(transformers.TrainerCallback):

    def on_step_end(self, args, state, control, **kwargs):
        if args.load_best_model_at_end:
            if (
                state.global_step <= state.max_steps * args.save_start_ratio
                or state.global_step <= args.save_start_steps
            ):
                control.should_save = False
        else:
            if (
                args.save_total_limit is not None
                and args.save_total_limit > 0
                and state.global_step <= state.max_steps - args.save_steps * args.save_total_limit
            ):
                control.should_save = False
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.best_metric is not None and state.best_model_checkpoint is not None:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.less if args.greater_is_better else np.greater
            if operator(metric_value, state.best_metric):
                control.should_save = False
        return control


class Main:

    def __init__(self):
        self.set_args()
        self.set_log()
        self.set_dir()
        self.set_fp32()
        transformers.set_seed(self.training_args.seed)

        config = transformers.AutoConfig.from_pretrained(
            self.model_args.model_name,
            cache_dir=self.model_args.cache_dir
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            cache_dir=self.model_args.cache_dir
        )

        model = getattr(import_module('model'), 'get_wsd_model')(type(config).__name__).from_pretrained(
            self.model_args.model_path if self.model_args.model_path else self.model_args.model_name,
            config=config,
            cache_dir=self.model_args.cache_dir
        )

        self.dataset_index = import_module(f'dataset.{self.data_args.dataset_index}')
        self.set_wsd_dataset()
        wsd_data_collator = getattr(self.dataset_index, 'DataCollatorForWsd')()

        self.trainer = transformers.Trainer(
            model=model,
            args=self.training_args,
            data_collator=wsd_data_collator,
            train_dataset=self.wsd_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=self.get_callbacks()
        )

    def set_args(self):
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, UserTrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()
        self.is_main_process = transformers.trainer_utils.is_main_process(self.training_args.local_rank)

    def make_dir(self, file):
        path = os.path.dirname(os.path.realpath(file))
        if not os.path.exists(path):
            os.makedirs(path)

    def set_dir(self):
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
            and len(os.listdir(self.training_args.output_dir)) > 0
        ):
            raise ValueError(f'Output directory ({self.training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
        if self.training_args.do_predict and self.is_main_process:
            self.make_dir(self.data_args.pred_file)

    def set_log(self):
        logging.basicConfig(
            format='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO if self.is_main_process else logging.WARN
        )
        if self.is_main_process:
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logger.warning(
            f'Process rank: {self.training_args.local_rank},'
            + f' device: {self.training_args.device},'
            + f' n_gpu: {self.training_args.n_gpu},'
            + f' distributed training: {bool(self.training_args.local_rank != -1)},'
            + f' 16-bits training: {self.training_args.fp16}'
        )
        logger.info(f'Training/evaluation parameters {self.training_args}')

    def set_fp32(self):
        if self.training_args.fp32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.warning(f'Use fp32 instead of tf32')

    def set_wsd_dataset(self):
        self.wsd_dataset = getattr(self.dataset_index, 'WsdDataset')(
            tokenizer=self.tokenizer,
            limit=self.data_args.input_limit,
            wsd_xml=self.data_args.wsd_xml,
            wsd_label=self.data_args.wsd_label,
            annotators=[0],
            is_dev=False,
            is_main_process=self.is_main_process
        )
        self.dev_dataset = getattr(self.dataset_index, 'WsdDataset')(
            tokenizer=self.tokenizer,
            limit=self.data_args.input_limit,
            wsd_xml=self.data_args.dev_xml,
            wsd_label=self.data_args.dev_label,
            annotators=[0, 1],
            is_dev=True,
            is_main_process=self.is_main_process
        ) if self.training_args.do_eval else None

    def compute_metrics(self, preds):
        preds, labels = preds
        segments = self.dev_dataset.get_segment_list()
        ok = 0
        for begin, end in zip(segments[:-1], segments[1:]):
            if preds[begin:end].argmax() < labels[begin:end].sum():
                ok += 1
        return {'eval_f1': ok / (len(segments) - 1) * 100}

    def get_callbacks(self):
        callbacks = []
        if self.training_args.early_stopping_patience > 0:
            callbacks.append(transformers.EarlyStoppingCallback(early_stopping_patience=self.training_args.early_stopping_patience))
        callbacks.append(UserFlowCallback)
        return callbacks

    def do_train(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.save_model()
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def do_predict(self):
        preds = self.trainer.predict(self.wsd_dataset).predictions
        examples = self.wsd_dataset.get_example_list()
        assert len(preds) == len(examples)
        with open(self.data_args.pred_file, 'w') as file:
            for (instance_id, lemma_key, *_), pred in zip(examples, preds):
                file.write('{},{},{}\n'.format(instance_id, lemma_key, pred))

    def run(self):
        if self.training_args.do_train:
            self.do_train()
        if self.training_args.do_predict:
            self.do_predict()


if __name__ == '__main__':
    Main().run()
